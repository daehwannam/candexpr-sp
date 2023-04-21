
'Tools for domain specific languages'

import re

from hissp.munger import munge, demunge

from dhnamlib.pylib.lisp import remove_comments, replace_prefixed_parens, parse_hy_args
from dhnamlib.pylib.iteration import merge_dicts, chainelems
from dhnamlib.pylib.function import starloop  # imported for eval_lissp
from dhnamlib.hissplib.macro import prelude
from dhnamlib.hissplib.compile import eval_lissp
from dhnamlib.hissplib.expression import remove_backquoted_symbol_prefixes

from .grammar import Action, MetaAction


prelude()  # used for eval_lissp


dsl_read_form = '(progn {})'


def read_dsl(file_path):
    def parse_kwargs(symbols):
        args, kwargs = parse_hy_args(symbols)
        assert len(args) == 0
        return dict([k.replace('-', '_'), v] for k, v in kwargs.items())

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
            return demunge(act_type)
        else:
            assert isinstance(act_type, (list, tuple))
            return tuple(map(demunge, demunge(act_type)))

    def make_define_types(super_types_dicts):
        def define_types(type_hierarchy_tuple):
            super_types_dict = {}

            def update_super_types_dict(parent_type, children):
                for child in children:
                    if isinstance(child, tuple):
                        child_kw, *grandchildren = child
                        assert child_kw[0] == ':'
                        child_type = child_kw[1:]
                        update_super_types_dict(child_type, grandchildren)
                    else:
                        child_type = demunge(child)
                    super_types_dict.setdefault(child_type, set()).add(parent_type)

            root_kw, *children = type_hierarchy_tuple
            root_type = root_kw[1:]
            update_super_types_dict(root_type, children)

            super_types_dicts.append(super_types_dict)
        return define_types

    def make_define_action(actions):
        def define_action(*symbols):
            kwargs = parse_kwargs(symbols)

            param_types, optional_idx, rest_idx = parse_params(kwargs['param_types'])

            kwargs.update(dict(
                name=demunge(kwargs['name']),
                act_type=parse_act_type(kwargs['act_type']),
                expr_dict=kwargs['expr_dict'],
                param_types=param_types,
                optional_idx=optional_idx,
                rest_idx=rest_idx))

            actions.append(Action(**kwargs))
        return define_action

    def make_define_meta_action(meta_actions):
        def define_meta_action(*symbols):
            kwargs = parse_kwargs(symbols)
            param_types, optional_idx, rest_idx = parse_params(kwargs['param_types'])
            param_types = param_types if len(param_types) != 0 else None
            act_type = parse_act_type(kwargs['act_type']) if 'act_type' in kwargs else None

            kwargs.update(dict(
                meta_name=demunge(kwargs['meta_name']),
                act_type=act_type,
                param_types=param_types,
                optional_idx=optional_idx,
                rest_idx=rest_idx))

            meta_actions.append(MetaAction(**kwargs))
        return define_meta_action

    def make_dict(*symbols):
        return parse_kwargs(symbols)

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

    def make_dsl(text, preprocessing_prefixed_parens=True):
        text = remove_comments(text)
        if preprocessing_prefixed_parens:
            text = preprocess_prefixed_parens(text)
        text = dsl_read_form.format(text)

        super_types_dicts = []
        actions = []
        meta_actions = []

        bindings = [['mapkv', make_dict],
                    ['define-types', make_define_types(super_types_dicts)],
                    ['define-action', make_define_action(actions)],
                    ['define-meta-action', make_define_meta_action(meta_actions)]]

        _ = eval_lissp(text, extra_ns=dict([munge(k), v] for k, v in bindings))
        assert _ is None

        dsl = dict(super_types_dict=merge_dicts(super_types_dicts,
                                                merge_fn=lambda values: set(chainelems(values))),
                   actions=actions,
                   meta_actions=meta_actions)

        return dsl

    with open(file_path) as f:
        text = f.read()

    return make_dsl(text)


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
    # dsl = read_dsl('./logic/example.dsl')
    dsl = read_dsl('./dsl/kopl/dsl')
    breakpoint()
    ()
