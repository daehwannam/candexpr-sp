
import os
from transformers import BartTokenizer, BartForConditionalGeneration


def is_finetuned(pretrained_model_name_or_path):
    return os.path.isfile(os.path.join(pretrained_model_name_or_path, '.finetuned'))


def load_tokenizer(pretrained_model_name_or_path, add_prefix_space, new_special_tokens=None, sorting_new_special_tokens=True):
    tokenizer = BartTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        add_prefix_space=add_prefix_space)
    if new_special_tokens is not None:
        assert not is_finetuned(pretrained_model_name_or_path)
        special_token_list = (sorted if sorting_new_special_tokens else list)(new_special_tokens)
        tokenizer.add_tokens(special_token_list, special_tokens=True)
    return tokenizer


def load_pretrained_model(pretrained_model_name_or_path, tokenizer=None):
    model = BartForConditionalGeneration(pretrained_model_name_or_path)
    if tokenizer is not None:
        assert not is_finetuned(pretrained_model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))
    return model


def forward_loss(model, batch):
    pass
