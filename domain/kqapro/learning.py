
import os
from itertools import chain
from typing import List, Tuple, Callable
import math
import warnings
import json
from fractions import Fraction

import torch
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import transformers
from configuration import config

from .execution import postprocess_prediction, invalid_program, get_counting_context

from splogic.formalism import InvalidCandidateActionError
from utility.trie import SpanTrie

from dhnamlib.pylib import filesys
from dhnamlib.pylib.hflib.transforming import logit_rescaling, MaskedLogitsProcessor
from dhnamlib.pylib import iteration
from dhnamlib.pylib.decoration import deprecated, construct
from dhnamlib.pylib.exception import NotFoundError
from dhnamlib.pylib.torchlib.dnn import (
    candidate_ids_to_mask, lengths_to_mask, masked_log_softmax, nll_without_reduction, pad_sequence)
from dhnamlib.pylib.data_structure import FIFODict


def is_finetuned(pretrained_model_name_or_path):
    return os.path.isfile(os.path.join(pretrained_model_name_or_path, '.finetuned'))


def load_tokenizer(pretrained_model_name_or_path, add_prefix_space, non_nl_tokens=None, sorting_non_nl_tokens=True):
    tokenizer = BartTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        add_prefix_space=add_prefix_space)
    if non_nl_tokens is not None:
        if is_finetuned(pretrained_model_name_or_path):
            assert all(token_id is not None for token_id in tokenizer.convert_tokens_to_ids(non_nl_tokens))
        else:
            # Example of addint new tokens:
            # - https://github.com/huggingface/transformers/issues/1413#issuecomment-538083512
            # - https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087
            # - https://github.com/huggingface/transformers/issues/3446#issuecomment-643171894

            ordered_non_nl_tokens = (sorted if sorting_non_nl_tokens else list)(non_nl_tokens)
            tokenizer.add_tokens(ordered_non_nl_tokens)

            # tokenizer.add_tokens(ordered_non_nl_tokens, special_tokens=True)
            # tokenizer.add_special_tokens(dict(additional_special_tokens=ordered_non_nl_tokens))  # https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087

    return tokenizer


def get_last_dir_path(model_learning_dir_path, dir_name='last'):
    return os.path.join(model_learning_dir_path, dir_name)


def get_best_dir_path(model_learning_dir_path, dir_name='best'):
    return os.path.join(model_learning_dir_path, dir_name)


def load_model(pretrained_model_name_or_path, num_tokens=None):
    model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    if num_tokens is None:
        assert is_finetuned(pretrained_model_name_or_path)
    else:
        if is_finetuned(pretrained_model_name_or_path):
            assert model.config.vocab_size == num_tokens
        else:
            model.resize_token_embeddings(num_tokens)

    # Not to make `NoRepeatNGramLogitsProcessor` in `GenerationMixin._get_logits_processor`
    model.config.no_repeat_ngram_size = None

    # Disable length_penalty for `BeamSearchScorer`
    model.config.length_penalty = 0

    # Disable early_stopping for `BeamSearchScorer`.
    # - `beam_search` with `early_stopping=True` results in worse performance than `greedy_search`.
    # - `early_stopping` stops `beam_search` when it find `self.num_beams` number of complete sequences regardless of their scores.
    #   Check `from transformers.generation_beam_search.BeamHypotheses.is_done`.
    model.config.early_stopping = False

    # 0 is the default value for MinLengthLogitsProcessor
    model.config.min_length = 0  

    return model


def load_model_config(pretrained_model_name_or_path):
    return BartConfig.from_pretrained(pretrained_model_name_or_path)


def get_param_groups(model, learning_rate, weight_decay):
    non_decayed_names = ["bias", "layernorm", "layer_norm"]
    model_named_parameters = tuple(model.named_parameters())

    def is_decayed_param_name(param_name):
        return not any(nd_name in param_name for nd_name in non_decayed_names)

    # 'weight_decay' is for AdamW
    # 'lr' values become LambdaLR's base_lrs which are multiplied by lr_lambdas
    param_groups = [
        dict(params=[param for param_name, param in model_named_parameters
                     if is_decayed_param_name(param_name)],
             weight_decay=weight_decay,
             lr=learning_rate),
        dict(params=[param for param_name, param in model_named_parameters
                     if not is_decayed_param_name(param_name)],
             weight_decay=0.0,
             lr=learning_rate)]

    return param_groups


def save_model(model, dir_path):
    filesys.touch_with_mkpdirs(os.path.join(dir_path, '.finetuned'))
    model.save_pretrained(dir_path)


def _token_id_seq_to_action_seq(grammar, token_id_seq):
    eos_token_id_idx = iteration.index(token_id_seq, grammar.lf_tokenizer.eos_token_id, reverse=True)
    assert token_id_seq[0] == grammar.lf_tokenizer.bos_token_id
    action_id_seq = token_id_seq[1: eos_token_id_idx]  # index 0 has bos_token_id, which should be skipped
    action_seq = tuple(map(grammar.id_to_action, action_id_seq))
    return action_seq


def token_id_seq_to_last_state(grammar, token_id_seq, ignoring_parsing_errors=False,
                               verifying=False, utterance_token_id_seq=None):
    try:
        action_seq = _token_id_seq_to_action_seq(grammar, token_id_seq)

        def get_last_state():
            return grammar.search_state_cls.get_last_state(action_seq, verifying=verifying)

        if utterance_token_id_seq is None:
            last_state = get_last_state()
        else:
            dynamic_trie = _utterance_token_id_seq_to_dynamic_trie(grammar, utterance_token_id_seq)
            with grammar.let_dynamic_trie(dynamic_trie):
                last_state = get_last_state()

        return last_state
    except NotFoundError:
        return grammar.get_invalid_state()
    except Exception as error:
        if ignoring_parsing_errors:
            return grammar.get_invalid_state()
        else:
            raise error

@deprecated
def _slow__labels_to_masks(grammar, labels):
    '''
    :param grammar:
    :param labels: a tensor of shape (batch_size, seq_len)
    '''
    assert labels.dim() == 2

    candidate_token_ids_seqs = []
    seq_lengths = []

    for token_id_seq in labels.tolist():
        action_seq = _token_id_seq_to_action_seq(grammar, token_id_seq)
        candidate_action_ids_seq = grammar.search_state_cls.action_seq_to_candidate_action_ids_seq(action_seq)
        candidate_token_ids_seq = list(chain([[grammar.lf_tokenizer.bos_token_id]],
                                             candidate_action_ids_seq,
                                             [[grammar.lf_tokenizer.eos_token_id]]))
        candidate_token_ids_seqs.append(candidate_token_ids_seq)
        seq_lengths.append(len(candidate_token_ids_seqs))  # including BOS and EOS

        # Note:
        # `len(candidate_action_ids_seq)` could be the length to either the last reduce action or EOS.
        # Actually, EOS token doesn't need to be considered,
        # since the probability of EOS after the last reduce action is always 1 during beam-search
        # due to logits_processor that ignores all actions except EOS.

    padded_candidate_token_ids_seqs = pad_sequence(candidate_token_ids_seqs, [grammar.lf_tokenizer.pad_token_id], dim=1)
    softmax_mask = candidate_ids_to_mask(padded_candidate_token_ids_seqs, len(grammar.lf_tokenizer))
    nll_mask = lengths_to_mask(seq_lengths, max_length=max(seq_lengths))
    # nll_mask and nll tensor have the same size

    return softmax_mask, nll_mask


def labels_to_masks(grammar, labels, utterance_token_ids, except_eos=False):
    '''
    :param grammar:
    :param labels: a tensor of shape (batch_size, seq_len)
    '''
    assert labels.dim() == 2

    allowed_and_ids_pairs_seqs = []
    seq_lengths = []

    for token_id_seq, utterance_token_id_seq in zip(labels.tolist(), utterance_token_ids.tolist()):
        action_seq = _token_id_seq_to_action_seq(grammar, token_id_seq)
        dynamic_trie = _utterance_token_id_seq_to_dynamic_trie(grammar, utterance_token_id_seq)
        with grammar.let_dynamic_trie(dynamic_trie):
            _allowed_and_ids_pairs_seq = grammar.search_state_cls.action_seq_to_allowed_and_ids_pairs_seq(action_seq)
        allowed_and_ids_pairs_seq = list(chain(
            [(True, [grammar.lf_tokenizer.bos_token_id])],
            _allowed_and_ids_pairs_seq,
            [(True, [grammar.lf_tokenizer.eos_token_id])]
        ))
        allowed_and_ids_pairs_seqs.append(allowed_and_ids_pairs_seq)
        seq_lengths.append(len(allowed_and_ids_pairs_seq))  # including BOS and EOS

        # Note:
        # `len(candidate_action_ids_seq)` could be the length to either the last reduce action or EOS.
        # Actually, EOS token doesn't need to be considered,
        # since the probability of EOS after the last reduce action is always 1 during beam-search
        # due to logits_processor that ignores all actions except EOS.

    padded_candidate_token_ids_seqs = pad_sequence(allowed_and_ids_pairs_seqs, (True, [grammar.lf_tokenizer.pad_token_id]), dim=1)
    softmax_mask = _allowed_and_ids_pairs_seqs_to_softmax_mask(padded_candidate_token_ids_seqs, len(grammar.lf_tokenizer))
    if except_eos:
        _seq_lengths = list(seq_length - 1 for seq_length in seq_lengths)
    else:
        _seq_lengths = seq_lengths
    nll_mask = lengths_to_mask(_seq_lengths, max_length=max(seq_lengths) + int(except_eos))
    # nll_mask and nll tensor have the same size

    return softmax_mask, nll_mask


def _allowed_and_ids_pairs_seqs_to_softmax_mask(allowed_and_ids_pairs_seqs, vocab_size):
    batch_size = len(allowed_and_ids_pairs_seqs)
    assert iteration.all_same(map(len, allowed_and_ids_pairs_seqs))
    seq_len = len(allowed_and_ids_pairs_seqs[0])

    softmax_mask = torch.full([batch_size, seq_len, vocab_size], fill_value=0, dtype=torch.int64)

    for idx_in_batch, allowed_and_ids_pairs_seq in enumerate(allowed_and_ids_pairs_seqs):
        for idx_in_seq, allowed_and_ids_pairs in enumerate(allowed_and_ids_pairs_seq):
            allowed, token_ids = allowed_and_ids_pairs
            if allowed:
                softmax_mask[idx_in_batch, idx_in_seq, token_ids] = 1
            else:
                softmax_mask[idx_in_batch, idx_in_seq, :] = 1
                softmax_mask[idx_in_batch, idx_in_seq, token_ids] = 0

    return softmax_mask


def _utterance_token_id_seq_to_dynamic_trie(grammar, utterance_token_id_seq):
    eos_token_id_idx = iteration.index(utterance_token_id_seq, grammar.utterance_tokenizer.eos_token_id, reverse=True)
    assert utterance_token_id_seq[0] == grammar.utterance_tokenizer.bos_token_id
    _utterance_token_id_seq = utterance_token_id_seq[1: eos_token_id_idx]
    first_utterance_token = grammar.utterance_tokenizer.convert_ids_to_tokens(_utterance_token_id_seq[0])
    if not first_utterance_token.startswith('Ġ'):
        _utterance_token_id_seq[0] = grammar.utterance_tokenizer.convert_tokens_to_ids('Ġ' + first_utterance_token)
    end_of_seq_id = grammar.reduce_token_id
    dynamic_trie = SpanTrie(_utterance_token_id_seq, end_of_seq_id)
    return dynamic_trie


def labels_to_nll_mask(grammar, labels, except_eos=False):
    '''
    :param grammar:
    :param labels: a tensor of shape (batch_size, seq_len)
    '''
    assert labels.dim() == 2

    seq_lengths = []

    for token_id_seq in labels.tolist():
        eos_token_id_idx = iteration.index(token_id_seq, grammar.lf_tokenizer.eos_token_id, reverse=True)
        seq_lengths.append(eos_token_id_idx + int(not except_eos))

    nll_mask = lengths_to_mask(seq_lengths, max_length=max(seq_lengths) + int(except_eos))
    # The "+1" of max_length is for EOS, so nll_mask and nll tensor have the same size

    return nll_mask


def compute_loss(grammar, logits, labels, softmax_mask=None, nll_mask=None):
    assert logits.dim() == 3
    assert labels.dim() == 2
    assert logits.size()[:-1] == labels.size()

    # softmax_mask, nll_mask = labels_to_masks(grammar, labels)

    log_probs = masked_log_softmax(logits, mask=softmax_mask, dim=-1)
    nll = nll_without_reduction(log_probs, labels)

    if nll_mask is None:
        masked_nll = nll
    else:
        masked_nll = nll * nll_mask.to(nll.device)

    loss = masked_nll.sum(dim=-1).mean(dim=0)

    return loss


@deprecated
def token_id_seq_to_program(grammar, token_id_seq):
    last_state = token_id_seq_to_last_state(grammar, token_id_seq)
    compiler = grammar.compiler_cls()
    program = compiler.compile_tree(last_state.tree)

    return program


def utterances_to_ids(grammar, utterances):
    encoded_utterances = grammar.utterance_tokenizer(utterances)
    utterance_token_ids = encoded_utterances['input_ids']
    return utterance_token_ids


@construct(tuple)
def generate_token_id_seqs(
        grammar, model, utterance_token_ids, max_length, num_beams,
        prefix_allowed_tokens_fn=None, logits_processor=transformers.LogitsProcessorList()
        # , **kwargs
):
    if logits_processor is None:
        logits_processor = transformers.LogitsProcessorList()

    # breakpoint()
    batched_output = model.generate(
        input_ids=utterance_token_ids,
        max_length=max_length,
        num_beams=num_beams,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor
        # **kwargs
    )

    batched_token_ids = batched_output[:, 1:]  # removing `decoder_start_token_id`

    for token_id_seq in batched_token_ids.tolist():
        try:
            eos_token_id_idx = iteration.index(token_id_seq, grammar.lf_tokenizer.eos_token_id, reverse=True)
            token_id_seq_without_padding = token_id_seq[:eos_token_id_idx + 1]
        except NotFoundError:
            token_id_seq_without_padding = token_id_seq
        yield token_id_seq_without_padding


def token_id_seqs_to_last_states(
        grammar, token_id_seqs, ignoring_parsing_errors=False, verifying=False,
        utterance_token_id_seqs=None):
    if utterance_token_id_seqs is None:
        _utterance_token_id_seqs = [None] * len(token_id_seqs)
    else:
        _utterance_token_id_seqs = utterance_token_id_seqs

    predicted_last_states = tuple(
        token_id_seq_to_last_state(
            grammar, token_id_seq, ignoring_parsing_errors=ignoring_parsing_errors,
            verifying=verifying, utterance_token_id_seq=utterance_token_id_seq)
        for token_id_seq, utterance_token_id_seq in zip(token_id_seqs, _utterance_token_id_seqs))

    return predicted_last_states


def last_states_to_programs(grammar, compiler, last_states, tolerant=False, ignoring_compilation_errors=False):
    def state_to_program(state):
        if grammar.is_invalid_state(state):
            return invalid_program
        else:
            if state.tree.is_closed_root():
                try:
                    return compiler.compile_tree(state.tree, tolerant=tolerant)
                except Exception as error:
                    if ignoring_compilation_errors:
                        return invalid_program
                    else:
                        raise error
            else:
                # when generation process reaches the max_length
                return invalid_program

    programs = tuple(state_to_program(last_state)
                     for last_state in last_states)
    return programs


@config
def programs_to_predictions(context, programs, max_num_program_iterations=config.ph):
    predictions = tuple(
        postprocess_prediction(program(get_counting_context(
            context, max_num_iterations=max_num_program_iterations)))
        for program in programs)

    return predictions


OPTIMIZER_FILE_NAME = 'optimizer.pt'
SCHEDULER_FILE_NAME = 'scheduler.pt'


def get_optimizer_file_path(dir_path):
    return os.path.join(dir_path, OPTIMIZER_FILE_NAME)


def save_optimizer(optimizer, dir_path):
    torch.save(optimizer.state_dict(), get_optimizer_file_path(dir_path))


def load_and_update_optimizer(optimizer, dir_path):
    optimizer.load_state_dict(torch.load(get_optimizer_file_path(dir_path)))


def get_scheduler_file_path(dir_path):
    return os.path.join(dir_path, SCHEDULER_FILE_NAME)


def save_scheduler(scheduler, dir_path):
    torch.save(scheduler.state_dict(), get_scheduler_file_path(dir_path))


def load_and_update_scheduler(scheduler, dir_path):
    scheduler.load_state_dict(torch.load(get_scheduler_file_path(dir_path)))


STATUS_FILE_NAME = 'status.json'


class StatusJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Fraction):
            return {'Fraction': (obj.numerator, obj.denominator)}
        else:
            return super().default(obj)


def as_python_object_for_status(dic):
    # https://stackoverflow.com/a/8230373
    if isinstance(dic, dict) and len(dic) == 1 and 'Fraction' in dic:
        return Fraction(*dic['Fraction'])
    else:
        return dic


def save_status(status, dir_path, file_name=STATUS_FILE_NAME):
    filesys.json_pretty_save(status, os.path.join(dir_path, file_name),
                             cls=StatusJSONEncoder)


def load_status(dir_path, file_name=STATUS_FILE_NAME):
    return filesys.json_load(os.path.join(dir_path, file_name),
                             object_hook=as_python_object_for_status)


class PerformanceJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Fraction):
            # return {'Fraction': (obj.numerator, obj.denominator)}
            # return repr(obj)
            return f'{obj.numerator}/{obj.denominator}'
        else:
            return super().default(obj)


def save_performance(performance, dir_path, file_name='performance.json'):
    updated_performance = dict(performance)
    updated_performance.update(accuracy_percent='{:5.2f}'.format(performance['accuracy'] * 100))
    filesys.json_pretty_save(updated_performance, os.path.join(dir_path, file_name),
                             cls=PerformanceJSONEncoder)


def save_analysis(analysis, dir_path, file_name='analysis.json'):
    analysis_file_path = os.path.join(dir_path, file_name)
    filesys.json_pretty_save(analysis, analysis_file_path)


def save_predictions(predictions, dir_path):
    predictions_file_path = os.path.join(dir_path, 'predictions.txt')
    filesys.write_lines(predictions_file_path, tuple(map(str, predictions)))


class SequencePrefixProcessor:
    def __init__(self, grammar, batch_size, num_beams, dynamic_tries, additional_mask_cache: dict = None):
        self.grammar = grammar
        self.dynamic_tries = dynamic_tries

        # multiplying "2" is for caching both previous states and the next sates
        self.num_beams = num_beams
        self.cache_size = batch_size * num_beams * 2
        self.state_fifo_dict = FIFODict(self.cache_size)

        self.DECODER_START_TOKEN_ID = grammar.model_config.decoder_start_token_id
        self.BOS_TOKEN_ID = grammar.lf_tokenizer.bos_token_id
        self.EOS_TOKEN_ID = grammar.lf_tokenizer.eos_token_id
        self.PAD_TOKEN_ID = grammar.lf_tokenizer.pad_token_id

        self.vocab_size = len(grammar.lf_tokenizer)
        self.additional_mask_cache = additional_mask_cache

    def action_id_seq_to_state(self, action_id_seq):
        assert isinstance(action_id_seq, tuple)
        curr_state = None

        if action_id_seq in self.state_fifo_dict:
            return self.state_fifo_dict[action_id_seq]
        else:
            if len(action_id_seq) == 0:
                curr_state = self.grammar.search_state_cls.create()
            else:
                if action_id_seq[:-1] in self.state_fifo_dict:
                    prev_state = self.state_fifo_dict[action_id_seq[:-1]]
                    if self.grammar.is_invalid_state(prev_state):
                        curr_state = self.grammar.get_invalid_state()
                    else:
                        next_action_id_seq = action_id_seq[-1:]  # a list with only the last element
                else:
                    warnings.warn('cache_size is not enough')
                    prev_state = None
                    next_action_id_seq = action_id_seq

                if curr_state is None:
                    try:
                        action_seq = tuple(map(self.grammar.id_to_action, next_action_id_seq))
                    except NotFoundError:
                        curr_state = self.grammar.get_invalid_state()
                    else:
                        try:
                            if prev_state.tree.is_closed_root():
                                # breakpoint()
                                assert self.num_beams > 1
                                curr_state = self.grammar.get_invalid_state()
                            else:
                                curr_state = self.grammar.search_state_cls.get_last_state(action_seq, initial_state=prev_state, verifying=True)
                        except InvalidCandidateActionError:
                            curr_state = self.grammar.get_invalid_state()

            self.state_fifo_dict[action_id_seq] = curr_state
            return curr_state

    def prefix_allowed_and_ids_pair_fn(self, batch_id: int, prefix_token_id_seq: torch.Tensor) -> List[int]:
        _prefix_token_id_seq = prefix_token_id_seq.tolist()

        # # Start of DEBUG
        # from .kopl_transfer import token_to_action_name
        # test_token_seq = ["<s>", "<count>", "<union>", "<filter-concept>", "<keyword-concept>", "Ġcounty", "Ġof", "ĠPennsylvania", "<reduce>", "<filter-number>", "<keyword-attribute-number>", "Ġpopulation", "<reduce>", "<constant-number>", "<constant-quantity>", "Ġ7", "800", "<reduce>", "<constant-unit>", "<reduce>", "<op-gt>", "<all-entities>", "<filter-concept>", "<keyword-concept>", "Ġcounty", "Ġof", "ĠPennsylvania", "<reduce>", "<filter-number>", "<keyword-attribute-number>", "Ġpopulation", "<reduce>", "<constant-number>"]
        # test_token_id_seq = [self.DECODER_START_TOKEN_ID, self.BOS_TOKEN_ID] + list(
        #     self.grammar.name_to_id(token_to_action_name(token, self.grammar.non_nl_tokens)) for token in test_token_seq[1:])
        # if test_token_id_seq == _prefix_token_id_seq:
        #     # breakpoint()
        #     pass
        # if test_token_id_seq == _prefix_token_id_seq[:-1]:
        #     if _prefix_token_id_seq[-1] == 50268:
        #         breakpoint()
        #         print(50268)
        #     if _prefix_token_id_seq[-1] == 3:
        #         breakpoint()
        #         print(3)
        # # End of DEBUG

        if len(_prefix_token_id_seq) == 1:
            # when `_prefix_token_id_seq` has only `self.DECODER_START_TOKEN_ID`
            return True, [self.BOS_TOKEN_ID]
        else:
            last_token_id = _prefix_token_id_seq[-1]
            if last_token_id in [self.PAD_TOKEN_ID, self.EOS_TOKEN_ID]:
                return True, [self.PAD_TOKEN_ID]
            else:
                decoder_start_token_id, bos_token_id, *action_id_seq = _prefix_token_id_seq
                assert decoder_start_token_id == self.DECODER_START_TOKEN_ID
                assert bos_token_id == self.BOS_TOKEN_ID
                with self.grammar.let_dynamic_trie(self.dynamic_tries[batch_id]):
                    curr_state = self.action_id_seq_to_state(tuple(action_id_seq))

                if self.grammar.is_invalid_state(curr_state):
                    return True, []
                elif curr_state.tree.is_closed_root():
                    return True, [self.EOS_TOKEN_ID]
                else:
                    with self.grammar.let_dynamic_trie(self.dynamic_tries[batch_id]):
                        return curr_state.get_allowed_and_ids_pairs()

    @deprecated
    def prefix_allowed_tokens_fn(self, batch_id: int, prefix_token_id_seq: torch.Tensor) -> List[int]:
        '''
        This function is passed to `transformers.PrefixConstrainedLogitsProcessor`, which is used by generate `transformers.generate`.

        :param batch_id: the index of an example in the input batch of search
        :param prefix_token_id_seq: a sequence of token ids
        :return: a list of allowed candidate token ids
        '''

        _prefix_token_id_seq = prefix_token_id_seq.tolist()

        if len(_prefix_token_id_seq) == 1:
            # when `_prefix_token_id_seq` has only `self.DECODER_START_TOKEN_ID`
            return [self.BOS_TOKEN_ID]
        else:
            # if len(_prefix_token_id_seq) >= 3:
            #     breakpoint()
            last_token_id = _prefix_token_id_seq[-1]
            if last_token_id in [self.PAD_TOKEN_ID, self.EOS_TOKEN_ID]:
                return [self.PAD_TOKEN_ID]
            else:
                decoder_start_token_id, bos_token_id, *action_id_seq = _prefix_token_id_seq
                assert decoder_start_token_id == self.DECODER_START_TOKEN_ID
                assert bos_token_id == self.BOS_TOKEN_ID
                curr_state = self.action_id_seq_to_state(tuple(action_id_seq))

                if self.grammar.is_invalid_state(curr_state):
                    return []
                elif curr_state.tree.is_closed_root():
                    return [self.EOS_TOKEN_ID]
                else:
                    return curr_state.get_candidate_action_ids()

    @deprecated
    def candidate_ids_to_mask(self, candidate_ids):
        return candidate_ids_to_mask(candidate_ids, self.vocab_size)

    @deprecated
    def prefix_to_mask_fn(self, batch_id: int, prefix_token_id_seq: torch.Tensor) -> List[int]:
        '''
        This function is passed to `dhnamlib.pylib.hflib.transforming.MaskedLogitsProcessor`,
        which is used by generate `transformers.generate`.

        :param batch_id: the index of an example in the input batch of search
        :param prefix_token_id_seq: a sequence of token ids
        :return: a list of allowed candidate token ids
        '''

        _prefix_token_id_seq = tuple(prefix_token_id_seq.tolist())

        def cache_mask(mask):
            self.additional_mask_cache[prefix_token_id_seq] = mask

        if _prefix_token_id_seq in self.additional_mask_cache:
            mask = self.additional_mask_cache[_prefix_token_id_seq]
        else:
            if len(_prefix_token_id_seq) == 1:
                # when `_prefix_token_id_seq` has only `self.DECODER_START_TOKEN_ID`
                mask = self.candidate_ids_to_mask([self.BOS_TOKEN_ID])
                cache_mask(mask)
            else:
                last_token_id = _prefix_token_id_seq[-1]
                if last_token_id in [self.PAD_TOKEN_ID, self.EOS_TOKEN_ID]:
                    mask = self.candidate_ids_to_mask([self.PAD_TOKEN_ID])
                    cache_mask(mask)
                else:
                    decoder_start_token_id, bos_token_id, *action_id_seq = _prefix_token_id_seq
                    assert decoder_start_token_id == self.DECODER_START_TOKEN_ID
                    assert bos_token_id == self.BOS_TOKEN_ID
                    curr_state = self.action_id_seq_to_state(tuple(action_id_seq))

                    if self.grammar.is_invalid_state(curr_state):
                        mask = self.candidate_ids_to_mask([])
                        cache_mask(mask)
                    elif curr_state.tree.is_closed_root():
                        mask = self.candidate_ids_to_mask([self.EOS_TOKEN_ID])
                        cache_mask(mask)
                    else:
                        # Not caching this mask
                        mask = curr_state.get_candidate_action_id_mask()
        return mask


@deprecated
def make_prefix_allowed_tokens_fn(grammar, batch_size, num_beams):
    sequence_prefix_processor = SequencePrefixProcessor(grammar, batch_size, num_beams)
    return sequence_prefix_processor.prefix_allowed_tokens_fn


@deprecated
def _get_rescaled_logits_processor(grammar, batch_size, num_beams):
    '''
    A logits processor with masked softmax.
    Don't use it if renormalizing sores is unnecessary.
    '''

    sequence_prefix_processor = SequencePrefixProcessor(grammar, batch_size, num_beams)
    prefix_allowed_tokens_fn = sequence_prefix_processor.prefix_allowed_tokens_fn
    logits_processor = transformers.LogitsProcessorList([
        logit_rescaling(transformers.PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams))])
    return logits_processor


_additional_mask_cache = dict()


@deprecated
def _old__get_logits_processor(grammar, batch_size, num_beams, renormalizing):
    '''logits processor for constrained decoding'''

    sequence_prefix_processor = SequencePrefixProcessor(
        grammar, batch_size, num_beams, additional_mask_cache=_additional_mask_cache)
    prefix_to_mask_fn = sequence_prefix_processor.prefix_to_mask_fn
    masked_logits_processor = MaskedLogitsProcessor(prefix_to_mask_fn, num_beams, renormalizing)
    logits_processor = transformers.LogitsProcessorList([masked_logits_processor])
    return logits_processor


def get_logits_processor(grammar, batch_size, num_beams, renormalizing, utterance_token_ids):
    '''logits processor for constrained decoding'''
    dynamic_tries = tuple(_utterance_token_id_seq_to_dynamic_trie(grammar, utterance_token_id_seq)
                          for utterance_token_id_seq in utterance_token_ids.tolist())
    sequence_prefix_processor = SequencePrefixProcessor(grammar, batch_size, num_beams, dynamic_tries)
    prefix_allowed_and_ids_pair_fn = sequence_prefix_processor.prefix_allowed_and_ids_pair_fn
    fast_prefix_constrained_logits_processor = FastPrefixConstrainedLogitsProcessor(
        prefix_allowed_and_ids_pair_fn, num_beams=num_beams)
    if renormalizing:
        fast_prefix_constrained_logits_processor = logit_rescaling(
            fast_prefix_constrained_logits_processor, postprocessing_nan=(num_beams > 1))
    logits_processor = transformers.LogitsProcessorList([fast_prefix_constrained_logits_processor])
    return logits_processor


class FastPrefixConstrainedLogitsProcessor(transformers.LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. This class is modified from `transformers.PrefixConstrainedLogitsProcessor`.

    Args: prefix_allowed_and_ids_pair_fn: (`Callable[[int, torch.Tensor], Tuple[bool, List[int]]]`): This
        function constraints the beam search to allowed tokens only at each step. This function takes 2
        arguments `inputs_ids` and the batch ID `batch_id`. It has to return a bool object and a token
        ids. When the bool object is True, the token ids should be allowed for the next generation step. When
        the bool object is False, the token ids should not be allowed for the next generation step. The bool
        object and toke ids are created conditioned on the previously generated tokens `inputs_ids` and the
        batch ID `batch_id`.

    """

    def __init__(self, prefix_allowed_and_ids_pair_fn: Callable[[int, torch.Tensor], Tuple[bool, List[int]]], num_beams: int):
        self._prefix_allowed_and_ids_pair_fn = prefix_allowed_and_ids_pair_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed, token_ids = self._prefix_allowed_and_ids_pair_fn(batch_id, sent)
                if allowed:
                    mask[batch_id * self._num_beams + beam_id, token_ids] = 0
                else:
                    mask[batch_id * self._num_beams + beam_id, :] = 0
                    mask[batch_id * self._num_beams + beam_id, token_ids] = -math.inf

        return scores + mask
