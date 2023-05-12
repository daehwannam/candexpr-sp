
import re
from itertools import chain

from transformers import BartTokenizer

from configuration import config
from logic.grammar import Grammar
from logic.formalism import Formalism, make_program_tree_cls, SearchState

from dhnamlib.pylib.decorators import cache
from dhnamlib.pylib.klass import Interface

from . import kopl_read
from . import kb_analysis

# from dhnamlib.pylib.decorators import fcache
from dhnamlib.pylib.hflib.transformers import get_all_non_special_tokens


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
def generate_act_type_trie_pairs(*, kb=config.ph, tokenizer, end_of_seq):
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


def get_special_token(action_name):
    return f'<{action_name}>'


def get_base_action_name(token):
    assert token[0] == '<'
    assert token[-1] == '>'
    return token[1:-1]

def get_nl_token_action_name(token):
    return f'(token {repr(token)})'

@config
def get_action_name(token, special_tokens=config.ph):
    if token in special_tokens:
        return get_base_action_name(token)
    else:
        return get_nl_token_action_name(token)


def register_all(register, grammar):
    @register(['function', 'join-tokens'])
    def join_tokens(tokens):
        return ''.join(tokens).replace('Ġ', ' ').lstrip()

    reduce_token = get_special_token(grammar.formalism.reduce_action.name)

    register(['name', 'token'], get_nl_token_action_name)

    def get_candidate_action_name(token):
        if token == reduce_token:
            return get_base_action_name(token)
        else:
            get_nl_token_action_name(token)

    def make_arg_filter(typ, is_prefix):
        def arg_filter(tree, actions):
            opened_tree, children = tree.get_opened_tree_children()
            for action in actions:
                action_seq = join_tokens(action_in_seq.meta_args[0] for action_in_seq in chain(
                    (child.value for child in children), [action]))
                if action == grammar.formalism.reduce_action:
                    yield kb_analysis.is_value_type(action_seq, typ)
                else:
                    yield is_prefix(action_seq)

        return arg_filter

    def get_act_type(typ):
        return f'v-{typ}'

    typ_is_prefix_pairs = [['quantity', kb_analysis.is_quantity_prefix],
                           ['date', kb_analysis.is_date_prefix],
                           ['year', kb_analysis.is_year_prefix]]

    for typ, is_prefix in typ_is_prefix_pairs:
        register(['filter', get_act_type(typ)], make_arg_filter(typ, is_prefix))

    def make_arg_candidate(trie):
        def arg_candidate(tree):
            opened_tree, children = tree.get_opened_tree_children()
            token_seq_prefix = tuple(child.value.meta_args[0] for child in children)
            candidate_tokens = trie.candidate_tokens(token_seq_prefix)

            return tuple(grammar.get_action(get_candidate_action_name(token)) for token in candidate_tokens)

        return arg_candidate

    for act_type, trie in generate_act_type_trie_pairs(tokenizer=grammar.tokenizer, end_of_seq=reduce_token):
        register(['filter', act_type], make_arg_candidate(trie))


@config
def make_tokenizer(special_tokens, pretrained_model_name_or_path=config.ph):
    tokenizer = BartTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        add_prefix_space=True)
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    return tokenizer


class KoPLGrammar(Grammar):
    interface = Interface(Grammar)

    @config
    def __init__(self, *, formalism, super_types_dict, actions, meta_actions, register=config.ph, use_reduce):
        super().__init__(formalism=formalism, super_types_dict=super_types_dict, actions=actions, meta_actions=meta_actions,
                         register=register, use_reduce=True)

        self.special_tokens = set(self._generate_special_tokens())
        self.tokenizer = make_tokenizer(self.special_tokens)

        def get_token_value_act_type_pairs():
            all_non_special_token_values = get_all_non_special_tokens(self.tokenizer)
            all_act_types = tuple(map(get_token_act_type, all_non_special_token_values))

            return tuple(zip(all_non_special_token_values, all_act_types))

        def iter_token_actions():
            nl_token_meta_action = self.get_meta_action('nl-token')
            for token_value, act_type in get_token_value_act_type_pairs():
                yield nl_token_meta_action(meta_args=dict(token=token_value),
                                           act_type=act_type)

        self.update_actions(iter_token_actions())
        register_all(register, self)

        self.kopl_function_to_action_dict = self._make_kopl_function_to_action_dict()

    @interface.implement
    def get_name_to_id_dicts(self):
        raise NotImplementedError

    @interface.implement
    def get_program_tree_cls(self):
        return make_program_tree_cls(self.formalism, name='KoPLProgramTree')

    @interface.implement
    def get_search_state_cls(self):
        raise NotImplementedError

    @interface.implement
    def get_compiler_cls(self):
        raise NotImplementedError

    def _generate_special_tokens(self):
        for action in self.base_actions:
            yield get_special_token(action.name)

    def _make_kopl_function_to_action_dict(self):
        kopl_function_to_action_dict = {}
        for action in self.base_actions:
            action_expr = action.expr_dict[self.formalism.default_expr_key]
            if isinstance(action_expr, str):
                kopl_function = kopl_read.extract_kopl_func(action_expr)
                if kopl_function is not None:
                    assert kopl_function not in kopl_function_to_action_dict
                    kopl_function_to_action_dict[kopl_function] = action
        return kopl_function_to_action_dict

    def _kopl_function_to_action(self, kopl_function):
        return self.kopl_function_to_action_dict.get(kopl_function)

    def _kopl_input_to_action(self, kopl_input):
        raise NotImplementedError('parse actions which are not kopl functions. (e.g. arithmetic operators)')
        raise NotImplementedError('use grammar.sub_and_super?')

    # @property
    # @cache
    # def search_state_cls(self):

    #     return KoPLSearchState

    def kopl_to_actions(self, labeled_kopl_program):
        actions = []
        recursive_kopl_form = kopl_read.kopl_to_recursive_form(labeled_kopl_program)

        def parse(form):
            # form['function']
            # form['inputs']
            for sub_form in form['dependencies']:
                pass

def make_kopl_search_state_cls(*, grammar, program_tree_cls):
    class KoPLSearchState(SearchState):
        interface = Interface(SearchState)

        @staticmethod
        @interface.implement
        def get_program_tree_cls():
            return program_tree_cls

        @interface.implement
        def get_initial_attrs(self):
            return dict()

        @interface.implement
        def get_start_action(self):
            return grammar.get_action('program')

        @interface.implement
        def get_updated_attrs(self, tree):
            return dict()

        @interface.implement
        def get_type_to_actions_dicts(self):
            return grammar.get_type_to_actions_dict()

        @interface.implement
        def get_name_to_id_dicts(self):
            return grammar.get_name_to_id_dicts()

    return KoPLSearchState



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
