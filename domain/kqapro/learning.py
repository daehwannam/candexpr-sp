
import os
from itertools import chain
from typing import List
import warnings
# import math

import torch
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import transformers
from configuration import config

from .execution import postprocess_prediction, invalid_program, get_counting_context

from logic.formalism import InvalidCandidateActionError

from dhnamlib.pylib import filesys
from dhnamlib.pylib.hflib.transformers import logit_rescaling
from dhnamlib.pylib import iteration
from dhnamlib.pylib.decoration import deprecated, construct
from dhnamlib.pylib.exception import NotFoundError
from dhnamlib.pylib.torchlib.dnn import candidate_ids_to_mask, lengths_to_mask, masked_log_softmax, nll_without_reduction, pad_sequence
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


def get_last_dir_path(model_learning_dir_path):
    return os.path.join(model_learning_dir_path, 'last')


def get_best_dir_path(model_learning_dir_path):
    return os.path.join(model_learning_dir_path, 'best')


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


def token_id_seq_to_last_state(grammar, token_id_seq):
    try:
        action_seq = _token_id_seq_to_action_seq(grammar, token_id_seq)
        last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=config.debug)

        return last_state
    except NotFoundError:
        return grammar.get_invalid_state()

def labels_to_masks(grammar, labels):
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
        seq_lengths.append(len(candidate_action_ids_seq))  # except couting EOS

        # Note:
        # `len(candidate_action_ids_seq)` could be the length to either the last reduce action or EOS.
        # Actually, EOS token doesn't need to be considered,
        # since the probability of EOS after the last reduce action is always 1 during beam-search
        # due to logits_processor that ignores all actions except EOS.

    padded_candidate_token_ids_seqs = pad_sequence(candidate_token_ids_seqs, [grammar.lf_tokenizer.pad_token_id], dim=1)
    softmax_mask = candidate_ids_to_mask(padded_candidate_token_ids_seqs, len(grammar.lf_tokenizer))
    nll_mask = lengths_to_mask(seq_lengths, max_length=max(seq_lengths) + 1)
    # The "+1" of max_length is for EOS, so nll_mask and nll tensor have the same size

    return softmax_mask, nll_mask


def labels_to_nll_mask(grammar, labels):
    '''
    :param grammar:
    :param labels: a tensor of shape (batch_size, seq_len)
    '''
    assert labels.dim() == 2

    seq_lengths = []

    for token_id_seq in labels.tolist():
        eos_token_id_idx = iteration.index(token_id_seq, grammar.lf_tokenizer.eos_token_id, reverse=True)
        seq_lengths.append(eos_token_id_idx)  # except couting EOS

    nll_mask = lengths_to_mask(seq_lengths, max_length=max(seq_lengths) + 1)
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


def token_id_seqs_to_last_states(grammar, token_id_seqs):
    predicted_last_states = tuple(token_id_seq_to_last_state(grammar, token_id_seq)
                                  for token_id_seq in token_id_seqs)

    return predicted_last_states


def last_states_to_programs(grammar, compiler, last_states, tolerant=False):
    def state_to_program(state):
        if grammar.is_invalid_state(state):
            return invalid_program
        else:
            if state.tree.is_closed_root():
                return compiler.compile_tree(state.tree, tolerant=tolerant)
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


def get_status_file_path(dir_path):
    return os.path.join(dir_path, STATUS_FILE_NAME)


def save_status(status, dir_path):
    filesys.json_pretty_save(status, get_status_file_path(dir_path))


def load_status(dir_path):
    return filesys.json_load(get_status_file_path(dir_path))


def save_analysis(analysis, dir_path):
    analysis_file_path = os.path.join(dir_path, 'analysis.json')
    filesys.json_pretty_save(analysis, analysis_file_path)


def save_predictions(predictions, dir_path):
    predictions_file_path = os.path.join(dir_path, 'predictions.txt')
    filesys.write_lines(predictions_file_path, predictions)


def make_prefix_allowed_tokens_fn(grammar, batch_size, num_beams):
    # multiplying "2" is for caching both previous states and the next sates
    cache_size = batch_size * num_beams * 2
    fifo_dict = FIFODict(cache_size)

    DECODER_START_TOKEN_ID = grammar.model_config.decoder_start_token_id
    BOS_TOKEN_ID = grammar.lf_tokenizer.bos_token_id
    EOS_TOKEN_ID = grammar.lf_tokenizer.eos_token_id
    PAD_TOKEN_ID = grammar.lf_tokenizer.pad_token_id

    def action_id_seq_to_state(action_id_seq):
        assert isinstance(action_id_seq, tuple)
        curr_state = None

        if action_id_seq in fifo_dict:
            return fifo_dict[action_id_seq]
        else:
            if len(action_id_seq) == 0:
                curr_state = grammar.search_state_cls.create()
            else:
                if action_id_seq[:-1] in fifo_dict:
                    prev_state = fifo_dict[action_id_seq[:-1]]
                    if grammar.is_invalid_state(prev_state):
                        curr_state = grammar.get_invalid_state()
                    else:
                        next_action_id_seq = action_id_seq[-1:]  # a list with only the last element
                else:
                    warnings.warn('cache_size is not enough')
                    prev_state = None
                    next_action_id_seq = action_id_seq

                if curr_state is None:
                    try:
                        action_seq = tuple(map(grammar.id_to_action, next_action_id_seq))
                    except NotFoundError:
                        curr_state = grammar.get_invalid_state()
                    else:
                        try:
                            curr_state = grammar.search_state_cls.get_last_state(action_seq, initial_state=prev_state, verifying=True)
                        except InvalidCandidateActionError:
                            curr_state = grammar.get_invalid_state()

            fifo_dict[action_id_seq] = curr_state
            return curr_state

    def prefix_allowed_tokens_fn(batch_id: int, prefix_token_id_seq: torch.Tensor) -> List[int]:
        '''
        This function is passed to `torch.PrefixConstrainedLogitsProcessor`, which is used by generate `transformers.generate`.

        :param batch_id: the index of an example in the input batch of search
        :param prefix_token_id_seq: a sequence of token ids
        :return: a list of allowed candidate token ids
        '''

        _prefix_token_id_seq = prefix_token_id_seq.tolist()

        if len(_prefix_token_id_seq) == 1:
            # when `_prefix_token_id_seq` has only `DECODER_START_TOKEN_ID`
            return [BOS_TOKEN_ID]
        else:
            # if len(_prefix_token_id_seq) >= 3:
            #     breakpoint()
            last_token_id = _prefix_token_id_seq[-1]
            if last_token_id in [PAD_TOKEN_ID, EOS_TOKEN_ID]:
                return [PAD_TOKEN_ID]
            else:
                decoder_start_token_id, bos_token_id, *action_id_seq = _prefix_token_id_seq
                assert decoder_start_token_id == DECODER_START_TOKEN_ID
                assert bos_token_id == BOS_TOKEN_ID
                curr_state = action_id_seq_to_state(tuple(action_id_seq))

                if grammar.is_invalid_state(curr_state):
                    return []
                if curr_state.tree.is_closed_root():
                    return [EOS_TOKEN_ID]
                else:
                    return curr_state.get_candidate_action_ids()

    return prefix_allowed_tokens_fn


def get_logits_processor(grammar, batch_size, num_beams):
    '''logits processor for masked softmax only'''

    prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(grammar, batch_size, num_beams)
    logits_processor = transformers.LogitsProcessorList([
        logit_rescaling(transformers.PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams))])
    return logits_processor
