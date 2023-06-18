
import os
from transformers import BartTokenizer, BartForConditionalGeneration


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
            assert model.vocab_size == num_tokens
        else:
            model.resize_token_embeddings(num_tokens)
    return model


def save_model(model):
    # TODO:
    # - touch '.finetuned'
    raise NotImplementedError


def forward_loss(model, batch):
    pass
