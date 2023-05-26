from itertools import chain

from transformers import BartTokenizer

from configuration import config
from logic.grammar import Grammar
from logic.formalism import make_program_tree_cls, make_search_state_cls

from dhnamlib.pylib.decorators import cache, construct, variable
from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.context import block
from dhnamlib.pylib.iteration import distinct_values
from dhnamlib.pylib.hflib.transformers import iter_id_token_pairs

from . import kb_analysis
from .compile import KoPLCompiler
from . import kopl_transfer

# from dhnamlib.pylib.decorators import fcache


class KoPLGrammar(Grammar):
    interface = Interface(Grammar)

    @config
    def __init__(self, *, formalism, super_types_dict, actions, start_action, meta_actions, register=config.ph, use_reduce=True):
        super().__init__(formalism=formalism, super_types_dict=super_types_dict, actions=actions, start_action=start_action,
                         meta_actions=meta_actions, register=register, use_reduce=use_reduce)

        self.non_nl_tokens = set(distinct_values(
            kopl_transfer.action_name_to_special_token(action.name)
            for action in self.base_actions))
        self.tokenizer = make_tokenizer(self.non_nl_tokens)
        register_all(register, self, self.tokenizer)
        self.update_actions(kopl_transfer.iter_nl_token_actions(
            self.meta_name_to_meta_action, self.tokenizer))

    @cache
    @interface.implement
    def get_name_to_id_dicts(self):
        @variable
        @construct(dict)
        def name_to_id_dict():
            for token_id, token in iter_id_token_pairs(self.tokenizer):
                action_name = kopl_transfer.token_to_action_name(token, special_tokens=self.non_nl_tokens)
                yield action_name, token_id

        return [name_to_id_dict]

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


@config
def make_tokenizer(special_tokens, pretrained_model_name_or_path=config.ph):
    tokenizer = BartTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        add_prefix_space=True)
    tokenizer.add_tokens(tuple(special_tokens), special_tokens=True)
    return tokenizer


def register_all(register, grammar, tokenizer):
    @register(['name', 'nl-token'])
    def get_nl_token_name(token):
        return kopl_transfer.nl_token_to_action_name(token)

    @register(['function', 'join-tokens'])
    def join_tokens(tokens):
        return ''.join(tokens).replace('Ġ', ' ').lstrip()

    with block:
        # filter
        def make_arg_filter(typ, is_prefix):
            def arg_filter(tree, actions):
                opened_tree, children = tree.get_opened_tree_children()
                for action in actions:
                    action_seq = join_tokens(action_in_seq.meta_args[0] for action_in_seq in chain(
                        (child.value for child in children), [action]))
                    if action == grammar.reduce_action:
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

    with block:
        # arg_candidate
        reduce_token = kopl_transfer.action_name_to_special_token(grammar.reduce_action.name)

        def get_candidate_action_name(token):
            return kopl_transfer.token_to_action_name(token, special_tokens=[reduce_token])

        def make_arg_candidate(trie):
            def arg_candidate(tree):
                opened_tree, children = tree.get_opened_tree_children()
                token_seq_prefix = tuple(child.value.get_meta_arg('token') for child in children)
                candidate_tokens = trie.candidate_tokens(token_seq_prefix)

                return tuple(grammar.name_to_action(get_candidate_action_name(token)) for token in candidate_tokens)

            return arg_candidate

        for act_type, trie in kopl_transfer.iter_act_type_trie_pairs(tokenizer=tokenizer, end_of_seq=reduce_token):
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
    from logic.grammar import read_grammar
    from dhnamlib.pylib.filesys import json_load
    from configuration import config
    from dhnamlib.pylib.time import TimeMeasure
    import cProfile

    grammar = read_grammar('./grammar/kopl/grammar.lissp', grammar_cls=KoPLGrammar)
    kb = json_load('./_tmp_data-indented/indented-kb.json')
    dataset = json_load('./_tmp_data-indented/indented-train.json')

    with config(kb=kb):
        action_seq = kopl_transfer.kopl_to_action_seq(grammar, dataset[0]['program'])
        with TimeMeasure() as tm:
            # last_seq = grammar.search_state_cls.get_last_state(action_seq, verifying=True)
            # last_seq = grammar.search_state_cls.get_last_state(action_seq, verifying=True)
            cProfile.run('last_seq = grammar.search_state_cls.get_last_state(action_seq, verifying=True)')
        print(f'Time: {tm.interval} seconds')

    # action_seq
    # breakpoint()
    pass
