
import re
from transformers import BartTokenizer


def is_digit(s):
    return re.match(r'^Ġ?[0-9]+$', s)


def analyze_tokens():
    tokenizer = BartTokenizer.from_pretrained('./pretrained/bart-base')
    all_tokens = tuple(map(tokenizer.convert_ids_to_tokens, range(tokenizer.vocab_size)))

    digit_tokens = tuple(filter(is_digit, all_tokens))
    print(len(digit_tokens))


if __name__ == '__main__':
    analyze_tokens()
