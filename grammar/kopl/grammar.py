
import re
import os

from transformers import BartTokenizer

from logic.grammar import Grammar
from .read import kopl_to_recursive_form

from dhnamlib.pylib.decorators import fcache


# Processing token actions
def is_digit_seq_token(token_value):
    return digit_seq_token_regex.match(token_value)


digit_seq_token_regex = re.compile(r'^Ġ?[0-9]+$')


def is_special_quantity_token(token_value):
    return token_value in {'.', '-', 'e'}


def is_special_date_token(token_value):
    return token_value in {'/'}


def is_special_year_token(token_value):
    return token_value in {'-'}


def get_token_act_type(token_value):
    is_digit_seq = is_digit_seq_token(token_value)
    is_special_quantity = is_special_quantity_token(token_value)
    is_special_year = is_special_year_token(token_value)
    is_special_date = is_special_date_token(token_value)

    types = ['vp-string']

    if is_digit_seq or is_special_quantity:
        types.append('vp-quantity')
    if is_digit_seq or is_special_date:
        types.append('vp-date')
    if is_digit_seq or is_special_year:
        types.append('vp-year')

    return types

# Processing types of keyword-part and vp-unit
def extract_type_value_pairs():
    pass


class KoPLGrammar(Grammar):
    def __init__(self, super_types_dict, actions, meta_actions):
        super().__init__(super_types_dict, actions, meta_actions)

        @fcache
        def get_token_value_act_type_pairs():
            tokenizer = BartTokenizer.from_pretrained('./pretrained/bart-base')
            all_token_values = tuple(map(tokenizer.convert_ids_to_tokens, range(tokenizer.vocab_size)))
            all_act_types = tuple(map(get_token_act_type, all_token_values))

            assert len(all_token_values) == len(all_act_types)

            return tuple(zip(all_token_values, all_act_types))

        def iter_token_actions():
            token_meta_action = self.get_meta_action('token')
            for token_value, act_type in get_token_value_act_type_pairs(
                    file_cache_path=os.path.join(os.path.dirname(__file__), 'cache__token_value_act_type_pairs.json')):
                yield token_meta_action(meta_args=dict(token=token_value),
                                        act_type=tuple(act_type))

        self.update_actions(iter_token_actions())


def kopl_to_actions(labeled_kopl_program):
    actions = []
    recursive_form = kopl_to_recursive_form(labeled_kopl_program)

    def parse(form):
        for sub_form in form['dependencies']:
            pass


#
# - A function
#   - No inputs, no dependencies -> terminal action
# - An input
#   - op-*
#     #+begin_src python
#     (('op-eq', '='), ('op-ne', '!='), ('op-lt', '<'), ('op-gt', '>'))
#     #+end_src
#   - direction-*
#     #+begin_src python
#     (('direction-forward', 'forward'), ('direction-backward', 'backward'))
#     #+end_src
#   - op-* 2
#     #+begin_src python
#     (('op-st', 'smaller'), ('op-gt', 'greater'))
#     #+end_src
#   - op-* 3
#     #+begin_src python
#     (('op-min', 'min'), ('op-max', 'max'))
#     #+end_src
#   - Others -> constants and tokens
#


# KB -> keyword-type pairs


if __name__ == '__main__':
    from logic.grammar import read_grammar
    grammar = read_grammar('./grammar/kopl/grammar.lissp')
    breakpoint()
    pass
