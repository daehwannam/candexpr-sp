
"Transferring from KoPL to Actions"

import re
from itertools import chain

from configuration import config

from dhnamlib.pylib.context import block
from dhnamlib.pylib.hflib.transformers import iter_default_non_special_tokens
from dhnamlib.pylib.decorators import construct, cache
from dhnamlib.pylib.function import compose
from dhnamlib.pylib.iteration import distinct_pairs, unique, merge_pairs, finditer

from . import kopl_read
from . import kb_analysis


def iter_nl_token_actions(meta_name_to_meta_action, tokenizer):
    nl_token_meta_action = meta_name_to_meta_action(nl_token_meta_name)

    def iter_token_value_act_type_pairs():
        all_non_special_token_values = tuple(iter_default_non_special_tokens(tokenizer))
        all_act_types = map(get_token_act_type, all_non_special_token_values)

        return zip(all_non_special_token_values, all_act_types)

    def iter_token_actions():
        for token_value, act_type in iter_token_value_act_type_pairs():
            yield nl_token_meta_action(meta_args=dict(token=token_value),
                                       act_type=act_type)

    return iter_token_actions()


nl_token_meta_name = 'nl-token'


with block:
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

        union_type = [
            'kp-concept', 'kp-entity', 'kp-relation',
            'kp-attr-string', 'kp-attr-number', 'kp-attr-time',
            'kp-q-string', 'kp-q-number', 'kp-q-time'
            'vp-string', 'vp_unit']

        if is_digit_seq or is_special_quantity:
            union_type.append('vp-quantity')
        if is_digit_seq or is_special_date:
            union_type.append('vp-date')
        if is_digit_seq or is_special_year:
            union_type.append('vp-year')

        return tuple(union_type)

    @config
    def iter_act_type_trie_pairs(*, kb=config.ph, tokenizer, end_of_seq):
        kb_info = kb_analysis.extract_kb_info(kb)
        # data_types = set(['string', 'quantity', 'time', 'date', 'year'])
        # time_types = set(['time', 'date', 'year'])

        key_to_act_type = dict(
            concepts='kw-concept',
            entities='kw-entity',
            relations='kw-relation',
            units='v-unit'
        )

        trie_dict = kb_analysis.make_trie_dict(kb_info, tokenizer, end_of_seq)
        for key, value in trie_dict.items():
            if isinstance(value, dict):
                pass
            else:
                trie = value
                yield key_to_act_type[key], trie

        data_type_to_attr_act_type = dict(
            string='kw-attr-string',
            quantity='kw-attr-number',
            time='kw-attr-time',
            date='kw-attr-time',
            year='kw-attr-time')
        data_type_to_q_act_type = dict(
            string='kw-q-string',
            quantity='kw-q-number',
            time='kw-q-time',
            date='kw-q-time',
            year='kw-q-time')

        assert sum(1 for value in trie_dict.values() if isinstance(value, dict)) == 2

        for data_type_to_act_type, data_type_to_trie in [[data_type_to_attr_act_type, trie_dict['data_type_to_attributes']],
                                                         [data_type_to_q_act_type, trie_dict['data_type_to_qualifiers']]]:
            for typ, trie in data_type_to_trie.items():
                yield data_type_to_act_type[typ], trie


with block:
    # Conversion bewteen tokens and action names
    def action_name_to_special_token(action_name):
        return f'<{action_name}>'

    def special_token_to_action_name(special_token):
        assert special_token[0] == '<'
        assert special_token[-1] == '>'
        return special_token[1:-1]

    _delimiter_after_prefix = ' '

    def _add_prefix(text, prefix):
        return f'{prefix}{_delimiter_after_prefix}{text}'

    def _remove_prefix(text_with_prefix, prefix):
        return text_with_prefix[len(prefix) + len(_delimiter_after_prefix):]

    def _add_parentheses(text):
        return f'({text})'

    def _remove_parentheses(text):
        assert text[0] == '('
        assert text[1] == ')'
        return text[1:-1]

    def nl_token_to_action_name(nl_token):
        return _add_parentheses(_add_prefix(nl_token, nl_token_meta_name))

    def action_name_to_nl_token(action_name):
        return _remove_parentheses(_remove_prefix(action_name, nl_token_meta_name))

    def is_nl_token_action_name(action_name):
        return action_name[0] == '(' and action_name[-1] == ')' \
            and _remove_parentheses(action_name).startswith(nl_token_meta_name + _delimiter_after_prefix)

    def token_to_action_name(token, special_tokens):
        if token in special_tokens:
            return special_token_to_action_name(token)
        else:
            return nl_token_to_action_name(token)

    def action_name_to_token(action_name):
        if is_nl_token_action_name(action_name):
            return action_name_to_nl_token(action_name)
        else:
            return action_name_to_special_token(action_name)


with block:
    def kopl_to_action_seq(grammar, labeled_kopl_program):
        action_seq = []
        recursive_kopl_form = kopl_read.kopl_to_recursive_form(labeled_kopl_program)

        def parse(form):
            action_seq.append(_kopl_function_to_action(grammar, form['function']))
            action_seq.extend(_kopl_input_to_action_seq(grammar, form['inputs']))
            for sub_form in form['dependencies']:
                parse(sub_form)

        parse(recursive_kopl_form)
        return action_seq

    def _kopl_function_to_action(grammar, kopl_function):
        return _get_kopl_function_to_action_dict(grammar)[kopl_function]

    @cache
    @construct(compose(dict, distinct_pairs))
    def _get_kopl_function_to_action_dict(grammar):
        # kopl_function_to_action_dict = {}
        for action in grammar.base_actions:
            action_expr = action.expr_dict[grammar.formalism.default_expr_key]
            if isinstance(action_expr, str):
                kopl_function = kopl_read.extract_kopl_func(action_expr)
                if kopl_function is not None:
                    yield kopl_function, action

    def _kopl_input_to_action_seq(grammar, kopl_input, act_type):
        def text_to_nl_token_actions(text):
            return tuple(
                chain(
                    map(compose(grammar.name_to_action, nl_token_to_action_name),
                        grammar.tokenizer.tokenizer(text)),
                    [grammar.reduce_action]))

        def is_type_of(super_type):
            return grammar.sub_and_super(act_type, super_type)

        if is_type_of('keyword'):
            keyword_action = unique(_act_type_to_actions(grammar, act_type))
            action_seq = tuple(chain([keyword_action], text_to_nl_token_actions(kopl_input)))
        elif is_type_of('operator'):
            candidate_actions = _act_type_to_actions(grammar, act_type)
            operator_action = unique(finditer(
                candidate_actions, kopl_input,
                test=lambda action, kopl_input: (kopl_input in action.expr_dict[grammar.formalism.default_expr_key])))
            action_seq = [operator_action]
        elif is_type_of('value'):
            value_action = unique(_act_type_to_actions(grammar, act_type))
            action_seq = [value_action]
            if is_type_of('v-number'):
                assert value_action.name == 'constant-number'
                quantity_part, unit_part = kopl_read.number_to_quantity_and_unit(kopl_input)
                # quantity
                action_seq.append(grammar.name_to_action('constant-quantity'))
                action_seq.extend(text_to_nl_token_actions(quantity_part))
                # unit
                action_seq.append(grammar.name_to_action('constant-unit'))
                action_seq.extend(text_to_nl_token_actions(unit_part))
            else:
                action_seq.extend(text_to_nl_token_actions(kopl_input))
        else:
            raise Exception(f'unexpected type "{act_type}"')

        return action_seq

    @cache
    def _get_act_type_to_actions_dict(grammar):
        return merge_pairs([action.act_type, action] for action in grammar.base_actions)

    def _act_type_to_actions(grammar, act_type):
        return _get_act_type_to_actions_dict(grammar)[act_type]
