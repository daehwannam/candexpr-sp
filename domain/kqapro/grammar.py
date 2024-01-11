
import copy
from itertools import chain
from functools import lru_cache

from configuration import config
from splogic.grammar import Grammar
from splogic.formalism import make_program_tree_cls, make_search_state_cls

from dhnamlib.pylib.decoration import construct, variable, unnecessary, deprecated
from dhnamlib.pylib.function import identity
from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.context import block, Environment
from dhnamlib.pylib.iteration import distinct_values
from dhnamlib.pylib.hflib.transforming import iter_id_token_pairs, join_tokens as _join_tokens
from dhnamlib.hissplib.expression import repr_as_hash_str
from dhnamlib.pylib.torchlib.dnn import candidate_ids_to_mask

from . import kb_analysis
from .execution import KoPLCompiler
from . import kopl_transfer
from . import learning

# from dhnamlib.pylib.decoration import fcache


class KoPLGrammar(Grammar):
    interface = Interface(Grammar)

    @config
    def __init__(self, *, formalism, super_types_dict, actions, start_action, meta_actions, register=config.ph,
                 is_non_conceptual_type=None, use_reduce=True,
                 inferencing_subtypes=config.ph(True), using_distinctive_union_types=config.ph(True),
                 using_spans_as_entities=config.ph(False),
                 pretrained_model_name_or_path=config.ph):

        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if not inferencing_subtypes:
            super_to_sub_actions = kopl_transfer.iter_super_to_sub_actions(super_types_dict, is_non_conceptual_type)
            actions = tuple(chain(actions, super_to_sub_actions))

        super().__init__(formalism=formalism, super_types_dict=super_types_dict, actions=actions, start_action=start_action,
                         meta_actions=meta_actions, register=register, is_non_conceptual_type=is_non_conceptual_type,
                         use_reduce=use_reduce, inferencing_subtypes=inferencing_subtypes)

        self.model_config = learning.load_model_config(pretrained_model_name_or_path)
        self.initialize_from_base_actions()
        self.using_spans_as_entities = using_spans_as_entities
        self.dynamic_scope = Environment(allowing_duplicate_candidate_ids=True)
        register_all(register, self, self.lf_tokenizer, self.dynamic_scope)
        self.add_actions(kopl_transfer.iter_nl_token_actions(
            self.meta_name_to_meta_action, self.lf_tokenizer, using_distinctive_union_types=using_distinctive_union_types))

    @lru_cache(maxsize=None)
    def initialize_from_base_actions(self):
        self.non_nl_tokens = set(distinct_values(
            kopl_transfer.action_name_to_special_token(action.name)
            for action in self.base_actions))

        # logical form tokenizer
        self.lf_tokenizer = learning.load_tokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            add_prefix_space=True,
            non_nl_tokens=self.non_nl_tokens)

        # utterance tokenizer
        with block:
            self.utterance_tokenizer = copy.copy(self.lf_tokenizer)
            self.utterance_tokenizer.add_prefix_space = False

    @lru_cache(maxsize=None)
    @interface.implement
    def get_name_to_id_dicts(self):
        self.initialize_from_base_actions()

        @variable
        @construct(dict)
        def name_to_id_dict():
            for token_id, token in iter_id_token_pairs(self.lf_tokenizer):
                action_name = kopl_transfer.token_to_action_name(token, special_tokens=self.non_nl_tokens)
                yield action_name, token_id

        return [name_to_id_dict]

    @lru_cache(maxsize=None)
    def _get_token_to_id_dict(self):
        self.initialize_from_base_actions()
        return dict(map(reversed, iter_id_token_pairs(self.lf_tokenizer)))

    def token_to_id(self, token):
        return self._get_token_to_id_dict()[token]

    @property
    @lru_cache(maxsize=None)
    def reduce_token(self):
        return kopl_transfer.action_name_to_special_token(self.reduce_action.name)

    @property
    @lru_cache(maxsize=None)
    def reduce_token_id(self):
        return self.lf_tokenizer.convert_tokens_to_ids(self.reduce_token)

    @lru_cache(maxsize=None)
    @interface.implement
    def get_program_tree_cls(self):
        return make_program_tree_cls(self.formalism, name='KoPLProgramTree')

    @config
    @lru_cache(maxsize=None)
    @interface.implement
    def get_search_state_cls(self, using_arg_candidate=config.ph, using_arg_filter=config.ph):
        @deprecated
        def ids_to_mask_fn(action_ids):
            return candidate_ids_to_mask(action_ids, len(self.lf_tokenizer))

        return make_search_state_cls(
            grammar=self,
            name='KoPLSearchState',
            using_arg_candidate=using_arg_candidate,
            using_arg_filter=using_arg_filter,
            ids_to_mask_fn=ids_to_mask_fn)

    @interface.implement
    def get_compiler_cls(self):
        return KoPLCompiler

    @interface.implement
    def iter_all_token_ids(self):
        return range(len(self.lf_tokenizer))

    # _INVALID_STATE = 'INVALID'

    # @classmethod
    # def is_invalid_state(cls, state):
    #     return state == cls._INVALID_STATE

    # @classmethod
    # def get_invalid_state(cls):
    #     return cls._INVALID_STATE

    def let_dynamic_trie(self, dynamic_trie, using_spans_as_entities=None):
        if using_spans_as_entities is None:
            using_spans_as_entities = self.using_spans_as_entities
        return self.dynamic_scope.let(dynamic_trie=dynamic_trie,
                                      using_spans_as_entities=using_spans_as_entities)


def register_all(register, grammar, lf_tokenizer, dynamic_scope):
    @register(['name', 'nl-token'])
    def get_nl_token_name(token):
        return kopl_transfer.nl_token_to_action_name(token)

    @register(['function', 'join-tokens'])
    def join_tokens(tokens):
        # return ''.join(tokens).replace('Ġ', ' ').lstrip()
        return _join_tokens(lf_tokenizer, tokens, skip_special_tokens=True).lstrip()

    def fast_join_tokens(tokens):
        '''
        Join tokens into a string. It's faster than `join_tokens`, but it
        cannot process tokens containing unicode characters.

        In the following example, this function doesn't work:

        >>> from transformers import BartTokenizer
        >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        >>> text = "Laos\u2013Vietnam border"
        >>> text
        'Laos–Vietnam border'
        >>> tokens = tokenizer.tokenize(text)
        >>> tokens
        ['L', 'aos', 'âĢĵ', 'V', 'iet', 'nam', 'Ġborder']
        >>> joined_tokens = fast_join_tokens(tokens)
        >>> joined_tokens
        'LaosâĢĵVietnam border'
        >>> text == joined_tokens
        False
        '''
        return ''.join(tokens).replace('Ġ', ' ').lstrip()

    @register(['function', 'concat-tokens'])
    def concat_tokens(*tokens):
        return join_tokens(tokens)

    @register(['function', 'concat-parts'])
    def concat_parts(*tokens):
        return repr_as_hash_str(join_tokens(tokens))

    @register(['function', 'concat-quantity-unit'])
    def concat_quantity_unit(quantity, unit):
        return repr_as_hash_str(f'{quantity} {unit}'.rstrip())

    reduce_token = grammar.reduce_token

    with block:
        # filter
        def make_arg_filter(typ, is_prefix):
            def arg_filter(tree, action_ids):
                opened_tree, children = tree.get_opened_tree_children()
                for action_id in action_ids:
                    action = grammar.id_to_action(action_id)
                    token_seq = [child.value.get_meta_arg('token') for child in children]
                    if action is grammar.reduce_action:
                        pass
                    else:
                        token_seq.append(action.get_meta_arg('token'))
                    joined_tokens = fast_join_tokens(token_seq)
                    if action == grammar.reduce_action:
                        if kb_analysis.is_value_type(joined_tokens, typ):
                            yield action_id
                    else:
                        if is_prefix(joined_tokens):
                            yield action_id
            return arg_filter

        def get_act_type(typ):
            return f'v-{typ}'

        typ_is_prefix_pairs = [['quantity', kb_analysis.is_quantity_prefix],
                               ['date', kb_analysis.is_date_prefix],
                               ['year', kb_analysis.is_year_prefix]]

        for typ, is_prefix in typ_is_prefix_pairs:
            register(['filter', get_act_type(typ)], make_arg_filter(typ, is_prefix))

    with block:
        # arg_candidate
        @unnecessary
        def get_candidate_action_name(token):
            return kopl_transfer.token_to_action_name(token, special_tokens=[reduce_token])

        def make_static_arg_candidate(trie):
            def arg_candidate(tree):
                opened_tree, children = tree.get_opened_tree_children()
                # token_seq_prefix = tuple(child.value.get_meta_arg('token') for child in children)
                # candidate_tokens = trie.candidate_tokens(token_seq_prefix)
                # return tuple(grammar.name_to_action(get_candidate_action_name(token)) for token in candidate_tokens)
                id_seq_prefix = tuple(grammar.token_to_id(child.value.get_meta_arg('token'))
                                      for child in children)
                return tuple(trie.candidate_ids(id_seq_prefix))
            return arg_candidate

        def make_entity_arg_candidate(static_trie):
            def arg_candidate(tree):
                opened_tree, children = tree.get_opened_tree_children()
                id_seq_prefix = tuple(grammar.token_to_id(child.value.get_meta_arg('token'))
                                      for child in children)
                # return tuple(static_trie.candidate_ids(id_seq_prefix, ignoring_errors=True))

                # # duplicate id may exists, but it's faster a little
                # return tuple(chain(dynamic_scope.dynamic_trie.candidate_ids(id_seq_prefix),
                #                    static_trie.candidate_ids(id_seq_prefix, ignoring_errors=True)))

                static_arg_candidate_ids = static_trie.candidate_ids(id_seq_prefix, ignoring_errors=True)

                if dynamic_scope.using_spans_as_entities:
                    if dynamic_scope.allowing_duplicate_candidate_ids:
                        # it's faster
                        preprocess = identity
                    else:
                        preprocess = set

                    arg_candidate_ids = tuple(preprocess(chain(
                        dynamic_scope.dynamic_trie.candidate_ids(id_seq_prefix),
                        static_arg_candidate_ids)))
                    del static_arg_candidate_ids
                else:
                    arg_candidate_ids = tuple(static_arg_candidate_ids)

                return arg_candidate_ids

            return arg_candidate

        for act_type, trie in kopl_transfer.iter_act_type_trie_pairs(lf_tokenizer=lf_tokenizer, end_of_seq=reduce_token):
            if act_type == 'kw-entity':
                arg_candidate = make_entity_arg_candidate(trie)
            else:
                arg_candidate = make_static_arg_candidate(trie)
            register(['candidate', act_type], arg_candidate)


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
    pass

