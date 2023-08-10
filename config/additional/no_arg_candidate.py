import argparse

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval


def _make_grammar():
    grammar = configuration._make_grammar()
    action_name = _get_action_name()
    if action_name is None:
        return grammar
    else:
        action = grammar.name_to_action(action_name)
        action.arg_candidate = None
        return grammar


def _get_action_name():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-arg-candidate-for', dest='action_name_without_arg_candidate', help='arg-candidate is disabled for the specified action')
    args, unknown = parser.parse_known_args()
    if args.action_name_without_arg_candidate == 'nothing':
        return None
    else:
        assert is_action_with_arg_candidate(args.action_name_without_arg_candidate)
        return args.action_name_without_arg_candidate


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
    ignoring_parsing_errors=True
)
