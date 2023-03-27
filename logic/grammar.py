from itertools import chain
from collections import deque
from typing import List, Dict
from abc import ABCMeta, abstractmethod
import copy
from bidict import bidict, ValueDuplicationError

from dhnamlib.pylib.structure import TreeStructure
from dhnamlib.pylib.iteration import any_not_none
from dhnamlib.pylib.decorators import abstractfunction
from dhnamlib.pylib.constant import Abstract


# token
class Action:
    def __init__(self,
                 *,
                 name,
                 act_type,
                 param_types,
                 expr_dict: Dict[str, List[str]],
                 optional_idx=None,
                 rest_idx=None):
        self.name = name

        assert isinstance(param_types, (list, tuple))

        self.act_type = act_type
        self.param_types = param_types
        self.expr_dict = expr_dict
        self.optional_idx = optional_idx
        self.rest_idx = rest_idx
        self.num_min_args = self.get_min_num_args()

    name_fn = None

    def __repr__(self):
        if self.name_fn is not None:
            assert self.name is None
            self.name_fn()
        else:
            return self.name

    def has_param(self):
        return bool(self.param_types)

    @property
    def num_params(self):
        return len(self.param_types)

    @property
    def terminal(self):
        return not self.param_types

    def get_min_num_args(self):
        additional_idx = any_not_none([self.optional_idx, self.rest_idx])
        if additional_idx is not None:
            return additional_idx + 1
        else:
            return len(self.param_types)


class MetaAction:
    def __init__(self,
                 *,
                 meta_name,
                 name_fn=None,
                 expr_dict_fn=None,
                 **action_kwargs):
        self.meta_name = meta_name
        meta_action = self

        class SpecificAction(Action):
            def __init__(self, *, meta_args, **kwargs):
                self.meta_action = meta_action

                for k, v in action_kwargs.items():
                    if k in kwargs:
                        assert v is None
                    else:
                        kwargs[k] = v
                assert 'name' not in kwargs
                kwargs['name'] = None

                assert 'expr_dict' not in kwargs
                kwargs['expr_dict'] = expr_dict_fn(meta_args)

                super().__init__(**kwargs)

        SpecificAction.name_fn = name_fn

        self.action_cls = SpecificAction

    def __repr__(self):
        return self.meta_name

    def __call__(self, *, meta_args, **kwargs):
        return self.action_cls(meta_args=meta_args, **kwargs)


class Grammar:
    def __init__(self, default_expr_key='default', placeholder_prefix='@'):
        self.default_expr_key = default_expr_key
        self.reduce_action = Action(name='reduce',
                                    act_type='<reduce>',
                                    param_types=[],
                                    expr_dict={self.default_expr_key: []})
        self.placeholder_prefix = placeholder_prefix

    @staticmethod
    def check_action_name_overlap(actions):
        # checking if a same name exists
        action_names = set()
        for action in actions:
            assert action.name not in action_names
            action_names.add(action.name)
        del action_names

    @staticmethod
    def make_name_to_action_dict(actions, constructor=dict):
        return constructor([action.name, action] for action in actions)

    @staticmethod
    def name_to_action(name, *name_to_action_dicts):
        return any(name_to_action_dict.get(name)
                   for name_to_action_dict in name_to_action_dicts)

    @staticmethod
    def make_type_to_actions_dict(actions, super_types_dict, constructor=dict):
        dic = {}
        for action in actions:
            type_q = deque([action.act_type])
            while type_q:
                typ = type_q.popleft()
                dic.setdefault(typ, set()).add(action)
                if typ in super_types_dict:
                    type_q.extend(super_types_dict[typ])
        if constructor == dict:
            return dic
        else:
            return constructor(dic.items())

    def _must_be_reduced(self, opened_tree, children):
        if len(children) > 0 and children[-1].value == self.reduce_action:
            return True
        elif opened_tree.value.rest_idx is not None:
            return False
        else:
            return len(children) == len(opened_tree.value.param_types)

    @staticmethod
    def _optionally_reducible(action, num_params):
        # it's called before an action is selected
        if action.rest_idx is not None:
            return num_params >= action.rest_idx
        else:
            assert num_params != action.num_params
            if action.optional_idx is not None:
                return num_params >= action.optional_idx
            else:
                return False

    def get_candidate_actions(self, opened_action, current_num_params, type_to_actions_dicts):
        rest_idx = opened_action.rest_idx
        if (rest_idx is not None) and (rest_idx < current_num_params):
            next_param_idx = rest_idx
        else:
            next_param_idx = current_num_params
        param_type = opened_action.param_types[next_param_idx]
        del rest_idx, next_param_idx

        return tuple(chain(*[type_to_actions_dict.get(param_type, [])
                             for type_to_actions_dict in type_to_actions_dicts],
                           [self.reduce_action] if Grammar._optionally_reducible(
                               opened_action, current_num_params) else []))

    def extend_actions(self, actions, use_reduce=True):
        if use_reduce:
            default_actions = tuple(chain([self.reduce_action], actions))
        else:
            default_actions = tuple(actions)
        return default_actions

    @staticmethod
    def make_name_to_id_dict(actions, start_id, constructor=dict, sorting=False):
        if sorting:
            names = sorted(set(action.name or action.name for action in actions))
        else:
            name_set = set()
            names = []
            for action in actions:
                if action.name not in name_set:
                    names.append(action.name)
                    name_set.add(action.name)
            assert len(name_set) == len(names)
        return constructor(map(reversed, enumerate(names, start_id)))

    @staticmethod
    def action_to_id(action, name_to_id_dicts):
        return any_not_none(name_to_id_dict.get(action.name or action.name)
                            for name_to_id_dict in name_to_id_dicts)


# program tree
class ProgramTree(TreeStructure, metaclass=ABCMeta):
    @classmethod
    def create_root(cls, value, terminal=False):
        # TODO: check this method. type of output object
        tree = super(ProgramTree, cls).create_root(value, terminal)
        tree.min_num_actions = 1 + value.num_min_args
        tree.cur_num_actions = 1
        return tree

    @staticmethod
    @abstractmethod
    def get_grammar():
        pass

    @property
    def grammar(self):
        return self.get_grammar()

    def push_action(self, action):
        opened, children = self.get_opened_tree_children()
        if opened.value.num_min_args <= len(children):
            addend = 1 + action.num_min_args
        else:
            addend = action.num_min_args

        new_tree = self.push_term(action) if action.terminal else self.push_nonterm(action)
        new_tree.min_num_actions = self.min_num_actions + addend

        new_tree.cur_num_actions = self.cur_num_actions + 1

        return new_tree

    def reduce_with_children(self, children, value=None):
        new_tree = super().reduce_with_children(children, value)
        new_tree.min_num_actions = children[-1].min_num_actions if children else self.min_num_actions
        new_tree.cur_num_actions = children[-1].cur_num_actions if children else self.cur_num_actions
        return new_tree

    def reduce_tree_amap(self):
        "reduce tree as much as possible"
        tree = self
        while not tree.is_closed_root():
            opened_tree, children = tree.get_opened_tree_children()
            if self.grammar._must_be_reduced(opened_tree, children):
                tree = opened_tree.reduce_with_children(children)
            else:
                break
        return tree

    def subtree_size_ge(self, size):
        stack = [self]
        accum = 1
        if accum >= size:
            return True

        while stack:
            tree = stack.pop()
            for child in reversed(tree.children):
                accum += 1
                if accum >= size:
                    return True
                if not tree.terminal:
                    assert not tree.opened
                    stack.append(child)

        return False

    def get_reduced_subtrees(self):
        tree = self
        subtrees = []
        if tree.is_closed():
            while not tree.terminal:
                assert not tree.opened
                subtrees.append(tree)
                tree = tree.children[-1]
        return subtrees

    def get_expr_str(self, expr_key=None):
        if expr_key is None:
            expr_key = self.grammar.default_expr_key

        def get_expr_pieces(action):
            return action.expr_dict.get(expr_key) or action.expr_dict.get(self.grammar.default_expr_key)
            return action.expr_pieces

        program_expr_pieces = []

        def tree_to_pieces(tree):
            for piece in get_expr_pieces(tree.value):
                if piece.startswith(self.grammar.placeholder_prefix):
                    param_idx = int(piece[1:])
                    tree_to_pieces(tree.children[param_idx])
                else:
                    program_expr_pieces.append(piece)

        tree_to_pieces(self)

        return ''.join(program_expr_pieces)


def make_program_tree_cls(grammar: Grammar, name=None):
    class NewProgramTree(ProgramTree):
        @staticmethod
        def get_grammar():
            return grammar

    if name is not None:
        NewProgramTree.__name__ = NewProgramTree.__qualname__ = name

    return NewProgramTree

# search state
class SearchState(metaclass=ABCMeta):
    @classmethod
    def create(cls):
        state = cls()
        state.initialize()
        return state

    @classmethod
    def get_grammar(cls):
        return cls.get_program_tree_cls().get_grammar()

    @property
    def grammar(self):
        return self.get_grammar()

    @staticmethod
    @abstractmethod
    def get_program_tree_cls():
        pass

    @property
    def program_tree_cls(self):
        return self.program_tree_cls()

    def initialize(self):
        self.tree = self.program_tree_cls.create_root(self.get_start_action())
        for k, v in self.get_initial_attrs().items():
            setattr(self, k, v)

    @abstractmethod
    def get_initial_attrs(self):
        pass

    @abstractmethod
    def get_start_action(self):
        pass

    def get_updated_tree(self, action):
        tree = self.tree.push_action(action).reduce_tree_amap()
        return tree

    def get_updated_state(self, tree):
        state = copy.copy(self)
        state.tree = tree
        for k, v in self.get_updated_attrs(tree).items():
            setattr(state, k, v)

        return state

    @abstractmethod
    def get_updated_attrs(self, tree):
        pass

    def _get_candidate_actions(self):
        opened_tree, children = self.tree.get_opened_tree_children()
        return self.grammar.get_candidate_actions(
            opened_tree.value, len(children), self.get_type_to_actions_dicts())

    @abstractmethod
    def get_type_to_actions_dicts(self):
        pass

    def _actions_to_ids(self, actions):
        name_to_id_dicts = self.get_name_to_id_dicts()

        def _action_to_id(action):
            return Grammar.action_to_id(action, name_to_id_dicts)

        return tuple(map(_action_to_id, actions))

    @abstractmethod
    def get_name_to_id_dicts(self):
        pass

    def get_candidate_action_to_id_bidict(self):
        actions = self._get_candidate_actions()
        ids = self._actions_to_ids(actions)

        assert len(actions) == len(ids)

        try:
            action_to_id_bidict = bidict(zip(actions, ids))
        except ValueDuplicationError as e:
            breakpoint()
            print(e)
        assert len(action_to_id_bidict) == len(actions)
        assert len(action_to_id_bidict.inverse) == len(ids)

        return action_to_id_bidict


def make_search_state_cls(program_tree_cls: type, name=None):
    class NewSearchState(SearchState):
        @staticmethod
        def get_program_tree_cls():
            return program_tree_cls

    if name is not None:
        NewSearchState.__name__ = NewSearchState.__qualname__ = name

    return NewSearchState


class DomainSpecificLanguage:
    num_all_ids = Abstract
    SearchState = Abstract

    @abstractfunction
    def compile_trees(self, trees):
        pass

    def compile_tree(self, tree):
        [executable] = self.compile_trees([tree])
        return executable
