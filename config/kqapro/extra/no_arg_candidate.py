import argparse
from functools import cache

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval


def _make_grammar():
    grammar = configuration._make_grammar()
    action_name = _get_action_name()
    if action_name in ['nothing', 'all']:
        return grammar
    else:
        action = grammar.name_to_action(action_name)
        action.arg_candidate = None
        return grammar


using_arg_candidate = True


@cache
def _get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-arg-candidate-for', dest='action_name_without_arg_candidate', help='arg-candidate is disabled for the specified action')
    args, unknown = parser.parse_known_args()

    return args


def _get_action_name():
    args = _get_cmd_args()
    if args.action_name_without_arg_candidate == 'nothing':
        return 'nothing'
    elif args.action_name_without_arg_candidate == 'all':
        # global using_arg_candidate
        # using_arg_candidate = False
        return 'all'
    else:
        assert is_action_with_arg_candidate(args.action_name_without_arg_candidate)
        return args.action_name_without_arg_candidate


def _get_using_arg_candidate():
    args = _get_cmd_args()
    if args.action_name_without_arg_candidate == 'all':
        return False
    else:
        return True


def is_action_with_arg_candidate(action_name):
    return action_name in _action_names


_action_names = set([
    'keyword-concept',
    'keyword-entity',
    'keyword-relation',
    'keyword-attribute-string',
    'keyword-attribute-number',
    'keyword-attribute-time',
    'keyword-qualifier-string',
    'keyword-qualifier-number',
    'keyword-qualifier-time',
    'constant-unit',
])


config = Environment(
    grammar=LazyEval(_make_grammar),
    ignoring_parsing_errors=True,
    using_arg_candidate=LazyEval(_get_using_arg_candidate),
)
