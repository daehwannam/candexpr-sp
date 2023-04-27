
from hissp.munger import munge, demunge

from dhnamlib.pylib.lisp import (remove_comments, replace_prefixed_parens, is_keyword, keyword_to_symbol)
from dhnamlib.pylib.iteration import merge_dicts, chainelems
from dhnamlib.pylib.function import starloop  # imported for eval_lissp
# from dhnamlib.pylib.structure import AttrDict
from dhnamlib.hissplib.macro import prelude
from dhnamlib.hissplib.compile import eval_lissp
from dhnamlib.hissplib.expression import remove_backquoted_symbol_prefixes  # imported for eval_lissp
from dhnamlib.hissplib.decorators import parse_hy_args, hy_function

from .formalism import Formalism, Action, MetaAction


prelude()  # used for eval_lissp

grammar_read_form = '(progn {})'


class Grammar:
    """Formal grammar"""

    def __init__(self, super_types_dict, actions, meta_actions, use_reduce=True):
        self.formalism = Formalism()

        self.super_types_dict = super_types_dict
        self.actions = self.formalism.extend_actions(actions, use_reduce=use_reduce)
        self.meta_actions = meta_actions

        self.name_to_action_dict = self.formalism.make_name_to_action_dict(actions)
        self.meta_name_to_meta_action_dict = self.formalism.make_name_to_action_dict(meta_actions, meta=True)

    def get_action(self, name):
        return self.formalism.name_to_action(name, [self.name_to_action_dict])

    def get_meta_action(self, meta_name):
        return self.formalism.name_to_action(meta_name, [self.meta_name_to_meta_action_dict])

    def add_action(self, action):
        assert action.name not in self.name_to_action_dict
        self.actions.append(action)
        self.name_to_action_dict[action.name] = action

    def update_actions(self, actions):
        for action in actions:
            self.add_action(action)

    # raise Exception('TODO: static_actions ...?')


def read_grammar(file_path, grammar_cls=Grammar):
    def parse_params(raw_params):
        munged_optional = munge('&optional')
        munged_rest = munge('&rest')

        optional_idx = None
        rest_idx = None
        params = []

        idx = 0
        for raw_param in raw_params:
            # if raw_param.endswith(munged_optional):
            if raw_param == munged_optional:
                assert optional_idx is None
                assert rest_idx is None
                optional_idx = idx
            # elif raw_param.endswith(munged_rest):
            elif raw_param == munged_rest:
                assert rest_idx is None
                rest_idx = idx
            else:
                params.append(demunge(raw_param))
                idx += 1

        return tuple(params), optional_idx, rest_idx

    def parse_act_type(act_type):
        if isinstance(act_type, str):
            return tuple([demunge(act_type)])
        else:
            assert isinstance(act_type, (list, tuple))
            return tuple(map(demunge, demunge(act_type)))

    def make_define_types(super_types_dicts, types):
        def define_types(type_hierarchy_tuple):
            super_types_dict = {}

            def update_super_types_dict(parent_type, children):
                for child in children:
                    if isinstance(child, tuple):
                        child_kw, *grandchildren = child
                        assert is_keyword(child_kw)
                        child_type = keyword_to_symbol(child_kw)
                        update_super_types_dict(child_type, grandchildren)
                    else:
                        assert isinstance(child, str)
                        child_type = demunge(child)
                    super_types_dict.setdefault(child_type, set()).add(parent_type)

            root_kw, *children = type_hierarchy_tuple
            root_type = root_kw[1:]
            update_super_types_dict(root_type, children)

            super_types_dicts.append(super_types_dict)

            for sub_type, super_types in super_types_dict.items():
                types.add(sub_type)
                types.update(super_types)

        return define_types

    def make_declare_abstract_types(abstract_types):
        def declare_abstract_types(types):
            for typ in map(demunge, types):
                if typ in abstract_types:
                    raise Exception(f'"{typ}" is declared more than once.')
                else:
                    abstract_types.add(typ)

        return declare_abstract_types

    def make_define_action(actions, is_concrete_type):
        def define_action(name, act_type, param_types, expr_dict, **kwargs):
            parsed_param_types, optional_idx, rest_idx = parse_params(param_types)
            act_type = parse_act_type(act_type)

            assert all(map(is_concrete_type, parsed_param_types))
            assert all(map(is_concrete_type, act_type))

            action = Action(
                name=demunge(name),
                act_type=act_type,
                expr_dict=expr_dict,
                param_types=parsed_param_types,
                optional_idx=optional_idx,
                rest_idx=rest_idx,
                **kwargs)

            actions.append(action)
        return define_action

    def make_define_meta_action(meta_actions, is_concrete_type):
        def define_meta_action(meta_name, *, act_type=None, param_types, **kwargs):
            parsed_param_types, optional_idx, rest_idx = parse_params(param_types)
            parsed_param_types = parsed_param_types if len(parsed_param_types) != 0 else None
            parsed_act_type = parse_act_type(act_type) if act_type is not None else None
                

            assert parsed_param_types is None or all(map(is_concrete_type, parsed_param_types))
            assert parsed_act_type is None or all(map(is_concrete_type, parsed_act_type))

            meta_action = MetaAction(
                meta_name=demunge(meta_name),
                act_type=parsed_act_type,
                param_types=parsed_param_types,
                optional_idx=optional_idx,
                rest_idx=rest_idx,
                **kwargs)

            meta_actions.append(meta_action)
        return define_meta_action

    def make_is_concrete_type(types, abstract_types):
        def is_concrete_type(typ):
            return (typ in types) and (typ not in abstract_types)

        return is_concrete_type

    def make_dict(*symbols):
        args, kwargs = parse_hy_args(symbols)
        return dict(*args, **kwargs)

    def preprocess_prefixed_parens(text):
        def expr_to_str(prefix, expr_repr):
            return 

        string_expr_prefix = '$'
        backquote = '`'
        return replace_prefixed_parens(
            text,
            info_dicts=[dict(prefix=string_expr_prefix, paren_pair='()',
                             fn=lambda x: '"{}{}"'.format(string_expr_prefix,
                                                          x.replace('"', r'\"'))),
                        dict(prefix=backquote, paren_pair='()',
                             fn=lambda x: "(remove_backquoted_symbol_prefixes {}{})".format(backquote, x))])

    def make_grammar(text, preprocessing_prefixed_parens=True):
        text = remove_comments(text)
        if preprocessing_prefixed_parens:
            text = preprocess_prefixed_parens(text)
        text = grammar_read_form.format(text)

        super_types_dicts = []
        types = set()
        abstract_types = set()
        actions = []
        meta_actions = []

        is_concrete_type = make_is_concrete_type(types, abstract_types)

        bindings = [['mapkv', make_dict],
                    ['define-types', make_define_types(super_types_dicts, types)],
                    ['declare-abstract-types', make_declare_abstract_types(abstract_types)],
                    ['define-action', hy_function(make_define_action(actions, is_concrete_type))],
                    ['define-meta-action', hy_function(make_define_meta_action(meta_actions, is_concrete_type))]]

        eval_result = eval_lissp(text, extra_ns=dict([munge(k), v] for k, v in bindings))
        assert eval_result is None

        grammar = grammar_cls(
            super_types_dict=merge_dicts(super_types_dicts,
                                         merge_fn=lambda values: set(chainelems(values))),
            actions=actions,
            meta_actions=meta_actions)

        return grammar

    with open(file_path) as f:
        text = f.read()

    return make_grammar(text)


def postprocess_denotation(denotation):
    if isinstance(denotation, list):
        new_denotation = [str(_) for _ in denotation]
    else:
        new_denotation = str(denotation)
    return new_denotation


def postprocess_answer(answer):
    if answer is None:
        new_answer = 'no'
    elif isinstance(answer, list) and len(answer) > 0:
        new_answer = answer[0]
    elif isinstance(answer, list) and len(answer) == 0:
        new_answer = 'None'
    else:
        new_answer = answer
    return new_answer


if __name__ == '__main__':
    # grammar = read_grammar('./logic/example.grammar')
    grammar = read_grammar('./grammar/kopl/grammar.lissp')
    breakpoint()
    ()
