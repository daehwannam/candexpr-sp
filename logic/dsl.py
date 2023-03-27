
'Tools for domain specific languages'

import re

from dhnamlib.pylib.lisp import remove_comments, preprocess_quotes, parse_hy_args
from dhnamlib.pylib.iteration import merge_dicts, chainelems
from dhnamlib.hissplib.macro import prelude
from dhnamlib.hissplib.compile import eval_lissp

from .grammar import Action


prelude()


split_lisp_regex = re.compile(r'([\(\)\[\] ])')


def split_lisp(text):
    tokens = split_lisp_regex.split(text)
    return tuple(token for token in tokens if token)


def split_expr_dict(expr_dict):
    return dict([k, split_lisp(v)]
                for k, v in expr_dict.items())


dsl_read_form = '(entuple {})'


def read_dsl(file_path):
    def parse_kwargs(symbols):
        args, kwargs = parse_hy_args(symbols)
        assert len(args) == 0
        return kwargs

    def make_action(*symbols):
        kwargs = parse_kwargs(symbols)
        kwargs['expr_dict'] = split_expr_dict(kwargs['expr_dict'])
        return dict(_action_=Action(**kwargs))

    def make_meta_action(*symbols):
        pass

    def make_super_types_dict(type_hierarchy_tuple):
        super_types_dict = {}

        def update_super_types_dict(parent_type, children):
            for child in children:
                if isinstance(child, tuple):
                    child_kw, *grandchildren = child
                    assert child_kw[0] == ':'
                    child_type = child_kw[1:]
                    update_super_types_dict(child_type, grandchildren)
                else:
                    child_type = child
                super_types_dict.setdefault(child_type, set()).add(parent_type)

        root_kw, *children = type_hierarchy_tuple
        root_type = root_kw[1:]
        update_super_types_dict(root_type, children)

        return dict(_types_=super_types_dict)

    def make_dict(*symbols):
        return parse_kwargs(symbols)

    def make_dsl(text):
        text = remove_comments(text)
        text = preprocess_quotes(text, round_to_string='$', square_to_round=True)
        text = dsl_read_form.format(text)

        def_list = eval_lissp(
            text, extra_ns=dict(defaction=make_action,
                                mapkv=make_dict,
                                deftypes=make_super_types_dict))

        merged_def_dict = merge_dicts(def_list)
        dsl = dict(actions=merged_def_dict['_action_'],
                   super_types_dict=merge_dicts(merged_def_dict['_types_'],
                                                merge_values=lambda values: set(chainelems(values))))
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
    dsl = read_dsl('./logic/example.dsl')
    breakpoint()
    ()
