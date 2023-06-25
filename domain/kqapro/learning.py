
import os
from itertools import chain

import torch
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration
from configuration import config

from dhnamlib.pylib import filesys
from dhnamlib.pylib.hflib.transformers import logit_rescaling
from dhnamlib.pylib import iteration
from dhnamlib.pylib.decoration import deprecated
from dhnamlib.pylib.torchlib.dnn import candidate_ids_to_mask, lengths_to_mask, masked_softmax, nll_without_reduction


def is_finetuned(pretrained_model_name_or_path):
    return os.path.isfile(os.path.join(pretrained_model_name_or_path, '.finetuned'))


def load_tokenizer(pretrained_model_name_or_path, add_prefix_space, non_nl_tokens=None, sorting_new_special_tokens=True):
    tokenizer = BartTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        add_prefix_space=add_prefix_space)
    if non_nl_tokens is not None:
        if is_finetuned(pretrained_model_name_or_path):
            assert all(token_id is not None for token_id in tokenizer.convert_tokens_to_ids(non_nl_tokens))
        else:
            special_token_list = (sorted if sorting_new_special_tokens else list)(non_nl_tokens)
            tokenizer.add_tokens(special_token_list, special_tokens=True)
    return tokenizer


def load_model(pretrained_model_name_or_path, num_tokens=None):
    model = BartForConditionalGeneration(pretrained_model_name_or_path)
    if num_tokens is not None:
        if is_finetuned(pretrained_model_name_or_path):
            assert model.config.vocab_size == num_tokens
        else:
            model.resize_token_embeddings(num_tokens)
    return model


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
    eos_token_id_idx = iteration.index(token_id_seq, grammar.lf_tokenizer.eos_token_id)
    action_id_seq = token_id_seq[:eos_token_id_idx]
    action_seq = tuple(map(grammar.id_to_action, action_id_seq))
    return action_seq


def token_id_seq_to_last_state(grammar, token_id_seq):
    action_seq = _token_id_seq_to_action_seq(grammar, token_id_seq)
    last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=config.debug)

    return last_state


def _labels_to_masks(grammar, labels):
    assert labels.dim() == 2

    candidate_token_ids_seqs = []
    seq_lengths = []

    for token_id_seq in labels:
        action_seq = _token_id_seq_to_action_seq(grammar, token_id_seq)
        candidate_action_ids_seq = grammar.search_state_cls.action_seq_to_candidate_action_ids_seq(action_seq)
        candidate_token_ids_seq = list(chain(candidate_action_ids_seq, [[grammar.lf_tokenizer.eos_token_id]]))
        candidate_token_ids_seqs.append(candidate_token_ids_seq)
        seq_lengths.append(len(candidate_action_ids_seq))  # except EOS
        # `len(candidate_action_ids_seq)` could be the length to either the last reduce action or EOS.
        # Actually, EOS token doesn't need to be considered,
        # since the probability of EOS after the last reduce action is always 1 during beam-search
        # due to logits_processor that ignores all actions except EOS.

    softmax_mask = candidate_ids_to_mask(candidate_token_ids_seqs, len(grammar.lf_tokenizer))
    nll_mask = lengths_to_mask(seq_lengths, max_length=max(seq_lengths) + 1)
    # The "+1" of max_length is for EOS, so nll_mask and nll tensor have the same size

    return softmax_mask, nll_mask


def compute_loss(grammar, logits, labels):
    assert logits.dim() == 3
    assert labels.dim() == 2
    assert logits.size()[:-1] == labels.size()

    softmax_mask, nll_mask = _labels_to_masks(grammar, labels)

    log_probs = masked_softmax(logits, mask=softmax_mask, dim=-1)
    nll = nll_without_reduction(log_probs, labels)

    masked_nll = nll * nll_mask
    loss = masked_nll.sum(dim=-1).mean(dim=0)

    return loss


@deprecated
def token_id_seq_to_program(grammar, token_id_seq):
    last_state = token_id_seq_to_last_state(grammar, token_id_seq)
    compiler = grammar.compiler_cls()
    program = compiler.compile_tree(last_state.tree)

    return program


def forward_loss(model, batch):
    pass


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


def get_logits_processor(grammar, batch_size, num_beams):
    prefix_allowed_tokens_fn = grammar.make_prefix_allowed_tokens_fn(batch_size, num_beams)
    logits_processor = torch.LogitsProcessorList([
        logit_rescaling(torch.PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams))])
    return logits_processor


def save_analysis(analysis, dir_path):
    analysis_file_path = os.path.join(dir_path, 'analysis.json')
    filesys.json_pretty_save(analysis, analysis_file_path)
