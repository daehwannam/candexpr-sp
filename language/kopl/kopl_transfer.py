
"Transferring from KoPL to Actions"

import re
from itertools import chain

from configuration import config
from util.trie import TokenTrie

from dhnamlib.pylib.context import block
from dhnamlib.pylib.hflib.transformers import iter_default_non_special_tokens
from dhnamlib.pylib.decorators import construct, cache, curry, variable, id_cache
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
            yield nl_token_meta_action(meta_kwargs=dict(token=token_value),
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
        # return token_value in {'/'}  # '/' is used in kb.json but not in kopl dataset
        return token_value in {'-'}

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
            'kp-q-string', 'kp-q-number', 'kp-q-time',
            'vp-string', 'vp_unit']

        if is_digit_seq or is_special_quantity:
            union_type.append('vp-quantity')
        if is_digit_seq or is_special_date:
            union_type.append('vp-date')
        if is_digit_seq or is_special_year:
            union_type.append('vp-year')

        return tuple(union_type)

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
        trie_dict['units'].add_token_seq([end_of_seq])  # add reduce-only case

        # Processing keywords in `key_to_act_type`
        for key, value in trie_dict.items():
            if isinstance(value, dict):
                pass
            else:
                trie = value
                yield key_to_act_type[key], trie

        assert sum(1 for value in trie_dict.values() if isinstance(value, dict)) == 2

        # Processing keywords for attributes and qualifiers
        def merge_tries(tries):
            if len(tries) > 1:
                return TokenTrie.merge(tries)
            else:
                return unique(tries)

        @variable
        @construct(curry(merge_pairs, merge_fn=merge_tries))
        def act_type_to_trie_dict():
            for data_type_to_act_type, data_type_to_trie in [[data_type_to_attr_act_type, trie_dict['type_to_attributes']],
                                                             [data_type_to_q_act_type, trie_dict['type_to_qualifiers']]]:
                for typ, trie in data_type_to_trie.items():
                    yield data_type_to_act_type[typ], trie

        yield from act_type_to_trie_dict.items()

    @config
    @id_cache
    def make_kw_to_type_dict(kb=config.ph):
        kb_info = kb_analysis.extract_kb_info(kb)
        _kw_to_type_dict = kb_analysis.make_kw_to_type_dict(kb_info)

        @construct(dict)
        def _make_kw_to_type_dict(kw_category, new_type_dict):
            for kw, typ in _kw_to_type_dict[kw_category].items():
                yield kw, new_type_dict[typ]

        return dict(
            [kw_category, _make_kw_to_type_dict(kw_category, new_type_dict)]
            for kw_category, new_type_dict in [['attribute', data_type_to_attr_act_type],
                                               ['qualifier', data_type_to_q_act_type]])


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
            function_action = _kopl_function_to_action(grammar, form['function'])
            action_seq.append(function_action)
            prev_kopl_input = None
            for idx, kopl_input in enumerate(form['inputs']):
                action_seq.extend(_kopl_input_to_action_seq(
                    grammar, kopl_input, function_action.param_types[idx],
                    prev_kopl_input=prev_kopl_input))
                prev_kopl_input = kopl_input
            for sub_form in form['dependencies']:
                parse(sub_form)

        parse(recursive_kopl_form)
        return action_seq

    def _kopl_function_to_action(grammar, kopl_function):
        if kopl_function == 'What':
            kopl_function = 'QueryName'
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

    @config
    def _kopl_input_to_action_seq(grammar, kopl_input, act_type, *, context=config.ph, prev_kopl_input):
        def text_to_nl_token_actions(text):
            return tuple(
                chain(
                    map(compose(grammar.name_to_action, nl_token_to_action_name),
                        grammar.tokenizer.tokenize(text)),
                    [grammar.reduce_action]))

        def is_type_of(act_type, super_type):
            return grammar.sub_and_super(act_type, super_type)

        kw_to_type_dict = make_kw_to_type_dict()

        if is_type_of(act_type, 'keyword'):
            if act_type in ['kw-attr-comparable', 'kw-attribute']:
                new_act_type = kw_to_type_dict['attribute'][kopl_input]
                if act_type == 'kw-attr-comparable':
                    assert new_act_type in ['kw-attr-number', 'kw-attr-time']
            elif act_type == 'kw-qualifier':
                new_act_type = kw_to_type_dict['qualifier'][kopl_input]
            else:
                new_act_type = act_type
            keyword_action = unique(_act_type_to_actions(grammar, new_act_type))
            action_seq = tuple(chain([keyword_action], text_to_nl_token_actions(kopl_input)))
        elif is_type_of(act_type, 'operator'):
            candidate_actions = _act_type_to_actions(grammar, act_type)
            if kopl_input == '=':
                operator_action = unique(action for action in candidate_actions if action.name == 'op-eq')
            else:
                operator_action = unique(finditer(
                    candidate_actions, kopl_input,
                    test=lambda action, kopl_input: (kopl_input in action.expr_dict[grammar.formalism.default_expr_key])))
            action_seq = [operator_action]
        elif is_type_of(act_type, 'value'):
            if act_type == 'value':
                new_act_type = 'v-{}'.format(kopl_read.classify_value_type(context, prev_kopl_input, kopl_input))
            else:
                new_act_type = act_type
            value_action = unique(_act_type_to_actions(grammar, new_act_type))
            action_seq = [value_action]
            if is_type_of(new_act_type, 'v-number'):
                assert value_action.name == 'constant-number'
                quantity_part, unit_part = kopl_read.number_to_quantity_and_unit(kopl_input)
                # quantity
                action_seq.append(grammar.name_to_action('constant-quantity'))
                action_seq.extend(text_to_nl_token_actions(quantity_part))
                # unit
                action_seq.append(grammar.name_to_action('constant-unit'))
                if unit_part == '1':
                    action_seq.extend([grammar.reduce_action])
                else:
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
