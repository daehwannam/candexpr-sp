import copy
from typing import List
import warnings

from configuration import config
from logic.grammar import Grammar
from logic.formalism import make_program_tree_cls, make_search_state_cls, InvalidCandidateActionError

from dhnamlib.pylib.decoration import cache, construct, variable, unnecessary, deprecated
from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.context import block
from dhnamlib.pylib.iteration import distinct_values, index, NotFoundError
from dhnamlib.pylib.hflib.transformers import iter_id_token_pairs, join_tokens as _join_tokens
from dhnamlib.pylib.data_structure import FIFODict
from dhnamlib.hissplib.expression import repr_as_hash_str

from . import kb_analysis
from .execution import KoPLCompiler
from . import kopl_transfer
from .model import is_finetuned, load_tokenizer

# from dhnamlib.pylib.decoration import fcache

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch


class KoPLGrammar(Grammar):
    interface = Interface(Grammar)

    @config
    def __init__(self, *, formalism, super_types_dict, actions, start_action, meta_actions, register=config.ph, use_reduce=True,
                 pretrained_model_name_or_path=config.ph):

        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        super().__init__(formalism=formalism, super_types_dict=super_types_dict, actions=actions, start_action=start_action,
                         meta_actions=meta_actions, register=register, use_reduce=use_reduce)

        self.initialize_from_base_actions()
        register_all(register, self, self.lf_tokenizer)
        self.add_actions(kopl_transfer.iter_nl_token_actions(
            self.meta_name_to_meta_action, self.lf_tokenizer))

    @cache
    def initialize_from_base_actions(self):
        self.non_nl_tokens = set(distinct_values(
            kopl_transfer.action_name_to_special_token(action.name)
            for action in self.base_actions))

        # logical form tokenizer
        with block:
            if is_finetuned(self.pretrained_model_name_or_path):
                load_tokenizer_kwargs = dict()
            else:
                load_tokenizer_kwargs = dict(
                    new_special_tokens=self.non_nl_tokens,
                    sorting_new_special_tokens=True)

            self.lf_tokenizer = load_tokenizer(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                add_prefix_space=True,
                **load_tokenizer_kwargs)

        # utterance tokenizer
        with block:
            self.utterance_tokenizer = copy.copy(self.lf_tokenizer)
            self.utterance_tokenizer.add_prefix_space = False

    @cache
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

    @cache
    def _get_token_to_id_dict(self):
        self.initialize_from_base_actions()
        return dict(map(reversed, iter_id_token_pairs(self.lf_tokenizer)))

    def token_to_id(self, token):
        return self._get_token_to_id_dict()[token]

    @cache
    @interface.implement
    def get_program_tree_cls(self):
        return make_program_tree_cls(self.formalism, name='KoPLProgramTree')

    @cache
    @interface.implement
    def get_search_state_cls(self):
        return make_search_state_cls(self, name='KoPLSearchState')

    @interface.implement
    def get_compiler_cls(self):
        return KoPLCompiler

    def make_prefix_allowed_tokens_fn(self, batch_size, num_beams):
        # multiplying "2" is for caching both previous states and the next sates
        cache_size = batch_size * num_beams * 2
        fifo_dict = FIFODict(cache_size)
        INVALID_STATE = 'INVALID'
        BOS_TOKEN_ID = self.lf_tokenizer.bos_token_id
        EOS_TOKEN_ID = self.lf_tokenizer.eos_token_id
        PAD_TOKEN_ID = self.lf_tokenizer.pad_token_id

        def action_id_seq_to_state(action_id_seq):
            assert isinstance(action_id_seq, tuple)
            curr_state = None

            if action_id_seq in fifo_dict:
                return fifo_dict[action_id_seq]
            else:
                if len(action_id_seq) == 0:
                    curr_state = self.search_state_cls.create()
                else:
                    if action_id_seq[:-1] in fifo_dict:
                        prev_state = fifo_dict[action_id_seq[:-1]]
                        if prev_state == INVALID_STATE:
                            curr_state = INVALID_STATE
                        else:
                            next_action_id_seq = action_id_seq[-1:]  # a list with only the last element
                    else:
                        warnings.warn('cache_size is not enough')
                        prev_state = None
                        next_action_id_seq = action_id_seq

                    if curr_state is None:
                        try:
                            action_seq = tuple(map(self.id_to_action, next_action_id_seq))
                        except NotFoundError:
                            curr_state = INVALID_STATE
                        else:
                            try:
                                curr_state = self.search_state_cls.get_last_state(action_seq, initial_state=prev_state, verifying=True)
                            except InvalidCandidateActionError:
                                curr_state = INVALID_STATE

                fifo_dict[action_id_seq] = curr_state
                return curr_state

        def prefix_allowed_tokens_fn(batch_id: int, prefix_token_id_seq: torch.Tensor) -> List[int]:
            '''
            This function is passed to `torch.PrefixConstrainedLogitsProcessor`, which is used by generate `transformers.generate`.

            :param batch_id: the index of an example in the input batch of search
            :param prefix_token_id_seq: a sequence of token ids
            :return: a list of allowed candidate token ids
            '''

            last_token_id = prefix_token_id_seq[-1]
            if last_token_id in [PAD_TOKEN_ID, EOS_TOKEN_ID]:
                return [PAD_TOKEN_ID]
            else:
                bos_token_id, *action_id_seq = prefix_token_id_seq
                assert bos_token_id == BOS_TOKEN_ID
                curr_state = action_id_seq_to_state(action_id_seq)

                if curr_state == INVALID_STATE:
                    return []
                if curr_state.tree.is_closed_root():
                    return [EOS_TOKEN_ID]
                else:
                    return curr_state.get_candidate_action_ids()

        return prefix_allowed_tokens_fn


def register_all(register, grammar, lf_tokenizer):
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

    reduce_token = kopl_transfer.action_name_to_special_token(grammar.reduce_action.name)

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

        def make_arg_candidate(trie):
            def arg_candidate(tree):
                opened_tree, children = tree.get_opened_tree_children()
                # token_seq_prefix = tuple(child.value.get_meta_arg('token') for child in children)
                # candidate_tokens = trie.candidate_tokens(token_seq_prefix)
                # return tuple(grammar.name_to_action(get_candidate_action_name(token)) for token in candidate_tokens)
                id_seq_prefix = tuple(grammar.token_to_id(child.value.get_meta_arg('token'))
                                      for child in children)
                return tuple(trie.candidate_ids(id_seq_prefix))

            return arg_candidate

        for act_type, trie in kopl_transfer.iter_act_type_trie_pairs(lf_tokenizer=lf_tokenizer, end_of_seq=reduce_token):
            if act_type == 'kw-entity':
                # KoPL doesn't consider valid entity names
                continue
            register(['candidate', act_type], make_arg_candidate(trie))


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

