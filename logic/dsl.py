
'Tools for domain specific languages'

import re
from itertools import chain

from dhnamlib.pylib.iteration import split_by_indices
from dhnamlib.pylib.lisp import parse_hy_args
from dhnamlib.hissplib.macro import prelude
from dhnamlib.hissplib.compile import eval_lissp

from .grammar import Action


prelude()


def get_quoted_paren_index_pairs(s, recursive=False):
    # https://stackoverflow.com/a/29992019

    '''
    e.g.
    >>> s = "(a b '(c d '(e f)))"
    >>> pairs = get_quoted_paren_index_pairs(s)
    >>> pair = pairs[0]
    >>> s[pair[0]: pair[1]]

    "(c d '(e f)"
    '''

    opening_stack = []  # stack of indices of opening parentheses
    pairs = []
    is_last_char_quote = False
    num_quoted_exprs_under_parsing = 0

    for i, c in enumerate(s):
        if c == "'":
            is_last_char_quote = True
        else:
            if c in ['(', '[']:
                opening_stack.append((i, c, is_last_char_quote))
                if is_last_char_quote:
                    num_quoted_exprs_under_parsing += 1
            elif c in [')', ']']:
                try:
                    idx, char, quoted = opening_stack.pop()
                    assert (char == '(' and c == ')') or (char == '[' and c == ']')
                    if quoted:
                        num_quoted_exprs_under_parsing -= 1
                        if recursive or num_quoted_exprs_under_parsing == 0:
                            pairs.append((idx, i))
                except IndexError:
                    print('Too many closing parentheses')
            is_last_char_quote = False
    if opening_stack:  # check if stack is empty afterwards
        print('Too many opening parentheses')

    return pairs


def preprocess_comments(text):
    splits = text.split('\n')
    # only remove lines that starts with comments
    return '\n'.join(split for split in splits if not split.lstrip().startswith(';'))


def preprocess_quotes(text):
    quoted_paren_index_pairs = get_quoted_paren_index_pairs(text)
    region_index_pairs = tuple((i - 1, j + 1)for i, j in quoted_paren_index_pairs)
    region_start_indices = set(start for start, end in region_index_pairs)
    region_indices = sorted(set(chain(*region_index_pairs)))
    splits = split_by_indices(text, region_indices)
    regions = []
    for start_idx, split in zip(chain([0], region_indices), splits):
        if start_idx in region_start_indices:
            if split.startswith("'("):
                # e.g. '(some symbols)
                region = '"{}"'.format(split[1:])
            else:
                # e.g. '[some symbols]
                assert split.startswith("'[")
                region = "'({})".format(split[2:-1])
        else:
            region = split
        regions.append(region)
    return ''.join(regions)


split_lisp_regex = re.compile(r'([\(\)\[\] ])')


def split_lisp(text):
    tokens = split_lisp_regex.split(text)
    return tuple(token for token in tokens if token)


def read_dsl(file_path):
    with open(file_path) as f:
        text = f.read()
        text = preprocess_comments(text)
        text = preprocess_quotes(text)
        text = '(entuple {})'.format(text)

    def parse_kwargs(symbols):
        args, kwargs = parse_hy_args(symbols)
        assert len(args) == 0
        return kwargs

    def make_action(*symbols):
        kwargs = parse_kwargs(symbols)
        kwargs['expr_dict'] = dict([k, split_lisp(v)]
                                   for k, v in kwargs['expr_dict'].items())
        return Action(**kwargs)

    def make_dict(*symbols):
        return parse_kwargs(symbols)

    actions = eval_lissp(text, extra_ns=dict(defaction=make_action, dict=make_dict))
    return actions
