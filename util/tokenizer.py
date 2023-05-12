
from transformers import BartTokenizer


# TODO
# - use Environment as config


def get_tokenizer(name_or_path, reduce_token):
    # name_or_path = './pretrained/bart-base'
    # reduce_token = '<reduce>'
    tokenizer = BartTokenizer.from_pretrained(
        name_or_path,
        add_prefix_space=True)
    tokenizer.add_tokens([reduce_token], special_tokens=True)
    return tokenizer
