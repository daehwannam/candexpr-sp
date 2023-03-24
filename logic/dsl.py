
'Tools for domain specific languages'

import re

from dhnamlib.pylib.lisp import remove_comments, preprocess_quotes, parse_hy_args
from dhnamlib.pylib.iteration import dicts2pairs
from dhnamlib.hissplib.macro import prelude
from dhnamlib.hissplib.compile import eval_lissp

from .grammar import Action


prelude()


split_lisp_regex = re.compile(r'([\(\)\[\] ])')


def split_lisp(text):
    tokens = split_lisp_regex.split(text)
    return tuple(token for token in tokens if token)


dsl_read_form = '(entuple {})'


def read_dsl(file_path):
    def parse_kwargs(symbols):
        args, kwargs = parse_hy_args(symbols)
        assert len(args) == 0
        return kwargs

    def make_action(*symbols):
        kwargs = parse_kwargs(symbols)
        kwargs['expr_dict'] = dict([k, split_lisp(v)]
                                   for k, v in kwargs['expr_dict'].items())
        return dict(action=Action(**kwargs))

    def make_type_hierarchy(type_hierarchy_tuple):
        parent_types_dict = {}

        def update_parent_types_dict(parent_type, children):
            for child in children:
                if isinstance(child, tuple):
                    child_kw, *grandchildren = child
                    child_type = child_kw[1:]
                    update_parent_types_dict(child_type, grandchildren)
                else:
                    child_type = child
                parent_types_dict.setdefault(child_type, set()).add(parent_type)

        root_kw, *children = type_hierarchy_tuple
        root_type = root_kw[1:]
        update_parent_types_dict(root_type, children)

        return dict(type=parent_types_dict)

    def make_dict(*symbols):
        return parse_kwargs(symbols)

    def make_dsl(text):
        text = remove_comments(text)
        text = preprocess_quotes(text, round_to_string=True, square_to_round=True)
        text = dsl_read_form.format(text)

        def_list = eval_lissp(
            text, extra_ns=dict(defaction=make_action,
                                dict=make_dict,
                                deftypes=make_type_hierarchy))

        types_defs = []
        action_defs = []

        for definition in def_list:
            for def_type, def_value in definition.items():
                if def_type == 'type':
                    types_defs.append(def_value)
                else:
                    assert def_type == 'action'
                    action_defs.append(def_value)

        def merge_parent_types_dicts(parent_types_dicts):
            new_parent_types_dict = {}
            for parent_types_dict in parent_types_dicts:
                for child, parents in parent_types_dict.items():
                    new_parent_types_dict.setdefault(child, set()).update(parents)
            return new_parent_types_dict

        dsl = dict(type_hierarchy=merge_parent_types_dicts(types_defs),
                   actions=action_defs)
        return dsl

    with open(file_path) as f:
        text = f.read()
        return make_dsl(text)


if __name__ == '__main__':
    dsl = read_dsl('./logic/example.dsl')
    breakpoint()
    ()
