import re
from itertools import chain
from collections import deque
from typing import List, Dict
from abc import ABCMeta, abstractmethod
import copy
from bidict import bidict, ValueDuplicationError
import inspect

from dhnamlib.pylib.structure import TreeStructure
from dhnamlib.pylib.iteration import any_not_none, all_same, flatten, split_by_indices, chainelems
from dhnamlib.pylib.klass import abstractfunction, Interface
from dhnamlib.pylib.decorators import unnecessary

from dhnamlib.hissplib.compile import eval_lissp
from dhnamlib.hissplib.macro import prelude


prelude()


# token
class Action:
    def __init__(self,
                 *,
                 name,
                 act_type,
                 param_types,
                 expr_dict,
                 optional_idx=None,
                 rest_idx=None,
                 arg_candidate=None,
                 arg_filter=None,
                 starting=False):
        self.name = name

        assert self.is_valid_act_type(act_type)
        assert isinstance(param_types, (list, tuple))

        self.act_type = act_type
        self.param_types = param_types
        self.expr_dict = expr_dict
        self.expr_pieces_dict = self.get_expr_pieces_dict(expr_dict)
        self.optional_idx = optional_idx
        self.rest_idx = rest_idx
        self.arg_candidate = arg_candidate
        self.arg_filter = arg_filter
        self.num_min_args = self.get_min_num_args()
        self.starting = starting

    _raw_left_curly_bracket_symbol = '___L_CURLY___'
    _raw_right_curly_bracket_symbol = '___R_CURLY___'
    _place_holder_regex = re.compile(r'{(([0-9]+)|([_a-zA-Z][_a-zA-Z0-9]*))}')

    class PieceKey:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return repr(self.value)

    @classmethod
    def get_expr_pieces_dict(cls, expr_dict):
        def replace_brackets_with_symbols(text):
            return text.replace(
                '{{', cls._raw_left_curly_bracket_symbol).replace(
                '}}', cls._raw_right_curly_bracket_symbol)

        def replace_symbols_with_brackets(text):
            return text.replace(
                cls._raw_left_curly_bracket_symbol, '{{').replace(
                cls._raw_right_curly_bracket_symbol, '}}')

        def iter_span_piece_key_pairs(expr):
            for match_obj in cls._place_holder_regex.finditer(expr):
                yield match_obj.span(), match_obj.group(1)

        def iter_expr_pieces(expr):
            span_piece_key_pairs = tuple(iter_span_piece_key_pairs(replace_brackets_with_symbols(expr)))
            if len(span_piece_key_pairs) == 0:
                yield expr
            else:
                spans, piece_keys = zip(*iter_span_piece_key_pairs(replace_brackets_with_symbols(expr)))
                splits = split_by_indices(expr, chainelems(spans))
                for idx, split in enumerate(splits):
                    if idx % 2 == 1:
                        yield Action.PieceKey(int(piece_keys[idx // 2]))
                    else:
                        yield replace_symbols_with_brackets(split)

        def get_expr_pieces(expr):
            if callable(expr):
                return expr
            else:
                assert isinstance(expr, str)
                return tuple(iter_expr_pieces(expr))

        return dict([piece_key, get_expr_pieces(expr)]
                    for piece_key, expr in expr_dict.items())

    @staticmethod
    def is_valid_act_type(act_type):
        if isinstance(act_type, tuple):
            return Action.is_valid_union_type(act_type)
        else:
            return isinstance(act_type, str)

    @staticmethod
    def is_valid_union_type(union_type):
        return len(union_type) > 1 and all(isinstance(typ, str) for typ in union_type)

    @staticmethod
    def is_union_type(typ):
        if isinstance(typ, tuple):
            assert Action.is_valid_union_type(typ)
            return True
        else:
            return False

    def is_union_act_type(self):
        return self.is_union_type(self.act_type)

    def __repr__(self):
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
                 num_meta_args=None,
                 name_fn,
                 expr_dict_fn,
                 **action_kwargs):

        self.num_meta_args = num_meta_args
        # assert all_same(map(self.get_func_num_args, [name_fn, expr_dict_fn]))
        # self.num_meta_args = self.get_func_num_args(name_fn)

        self.meta_name = meta_name
        meta_action = self

        class SpecificAction(Action):
            def __init__(self, *, meta_args, **kwargs):
                if meta_action.num_meta_args is not None:
                    assert len(meta_args) == meta_action.num_meta_args

                self.meta_args = meta_args

                self.meta_action = meta_action

                for k, v in action_kwargs.items():
                    if k in kwargs:
                        assert v is None
                    kwargs[k] = v

                assert 'name' not in kwargs
                kwargs['name'] = name_fn(*meta_args)

                assert 'expr_dict' not in kwargs
                kwargs['expr_dict'] = expr_dict_fn(*meta_args)

                super().__init__(**kwargs)

        self.action_cls = SpecificAction

    @unnecessary
    @staticmethod
    def get_func_num_args(func):
        return len(inspect.signature(func).parameters)

    def __repr__(self):
        return self.meta_name

    def __call__(self, *, meta_args, **kwargs):
        return self.action_cls(meta_args=meta_args, **kwargs)


class Formalism:
    """Formalism"""

    def __init__(self, default_expr_key='default'):
        self.default_expr_key = default_expr_key
        self.reduce_action = Action(name='reduce',
                                    act_type='reduce-type',
                                    param_types=[],
                                    expr_dict={self.default_expr_key: ''})

    @staticmethod
    def check_action_name_overlap(actions, meta=False):
        """Check if a same name exists"""

        attr = 'meta_name' if meta else 'name'
        action_names = set()
        for action in actions:
            assert getattr(action, attr) not in action_names
            action_names.add(getattr(action, attr))
        del action_names

    @staticmethod
    def make_name_to_action_dict(actions, constructor=dict, meta=False):
        dic = {}
        Formalism.update_name_to_action_dict(dic, actions, meta=meta)
        if isinstance(dic, constructor):
            return dic
        else:
            return constructor(dic.items())

    @staticmethod
    def update_name_to_action_dict(name_to_action_dict, actions, meta=False):
        attr = 'meta_name' if meta else 'name'
        for action in actions:
            if not getattr(action, attr) not in name_to_action_dict:
                breakpoint()
            assert getattr(action, attr) not in name_to_action_dict
            name_to_action_dict[getattr(action, attr)] = action

    @staticmethod
    def name_to_action(name, *name_to_action_dicts):
        action = any(name_to_action_dict.get(name)
                     for name_to_action_dict in name_to_action_dicts)
        assert action is not None
        return action

    @staticmethod
    def make_type_to_actions_dict(actions, super_types_dict, constructor=dict):
        dic = {}
        Formalism.update_type_to_actions_dict(dic, actions, super_types_dict)
        if isinstance(dic, constructor):
            return dic
        else:
            return constructor(dic.items())

    @staticmethod
    def update_type_to_actions_dict(type_to_actions_dict, actions, super_types_dict):
        for action in actions:
            type_q = deque(action.act_type if action.is_union_act_type() else
                           [action.act_type])
            while type_q:
                typ = type_q.popleft()
                type_to_actions_dict.setdefault(typ, set()).add(action)
                if typ in super_types_dict:
                    type_q.extend(super_types_dict[typ])

    @staticmethod
    def sub_and_super(super_types_dict, sub_type, super_type):
        type_q = deque(sub_type if Action.is_union_type(sub_type) else
                       [sub_type])
        assert not Action.is_union_type(super_type)

        while type_q:
            typ = type_q.popleft()
            if typ == super_type:
                return True
            else:
                if typ in super_types_dict:
                    type_q.extend(super_types_dict[typ])
        else:
            return False

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

        return tuple(chain(chainelems(type_to_actions_dict.get(param_type, [])
                                      for type_to_actions_dict in type_to_actions_dicts),
                           [self.reduce_action] if Formalism._optionally_reducible(
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
    def get_formalism():
        pass

    @property
    def formalism(self):
        return self.get_formalism()

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
            if self.formalism._must_be_reduced(opened_tree, children):
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
            expr_key = self.formalism.default_expr_key

        def get_expr_pieces(action):
            return any_not_none(
                action.expr_pieces_dict.get(k)
                for k in [expr_key, self.formalism.default_expr_key])

        def get_expr_form(tree):
            child_expr_forms = tuple(map(get_expr_form, tree.children))
            expr_pieces_or_expr_fn = get_expr_pieces(tree.value)
            if callable(expr_pieces_or_expr_fn):
                expr_fn = expr_pieces_or_expr_fn
                expr_form = expr_fn(*map(form_to_str, child_expr_forms))
            else:
                expr_pieces = expr_pieces_or_expr_fn
                expr_form = []
                for expr_piece in expr_pieces:
                    if isinstance(expr_piece, Action.PieceKey):
                        expr_form.append(child_expr_forms[expr_piece.value])
                    else:
                        expr_form.append(expr_piece)
            return expr_form

        def form_to_str(expr_form):
            return ''.join(flatten(expr_form))

        return form_to_str(get_expr_form(self))


def make_program_tree_cls(formalism: Formalism, name=None):
    class NewProgramTree(ProgramTree):
        interface = Interface(ProgramTree)

        @interface.implement
        @staticmethod
        def get_formalism():
            return formalism

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
    def get_formalism(cls):
        return cls.get_program_tree_cls().get_formalism()

    @property
    def formalism(self):
        return self.get_formalism()

    @staticmethod
    @abstractmethod
    def get_program_tree_cls():
        pass

    @property
    def program_tree_cls(self):
        return self.get_program_tree_cls()

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
        opened_action = opened_tree.value
        if opened_action.arg_candidate is None:
            actions = self.formalism.get_candidate_actions(
                opened_action, len(children), self.get_type_to_actions_dicts())
        else:
            actions = opened_action.arg_candidate(self.tree)
        if opened_action.arg_filter is not None:
            actions = tuple(opened_action.arg_filter(self.tree, actions))
        return actions

    @abstractmethod
    def get_type_to_actions_dicts(self):
        pass

    def _actions_to_ids(self, actions):
        name_to_id_dicts = self.get_name_to_id_dicts()

        def _action_to_id(action):
            return Formalism.action_to_id(action, name_to_id_dicts)

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
            raise e
        assert len(action_to_id_bidict) == len(actions)
        assert len(action_to_id_bidict.inverse) == len(ids)

        return action_to_id_bidict


def make_search_state_cls(grammar, name=None):
    class BasicSearchState(SearchState):
        interface = Interface(SearchState)

        @staticmethod
        @interface.implement
        def get_program_tree_cls():
            return grammar.program_tree_cls

        @interface.implement
        def get_initial_attrs(self):
            return dict()

        @interface.implement
        def get_start_action(self):
            return grammar.start_action

        @interface.implement
        def get_updated_attrs(self, tree):
            return dict()

        @interface.implement
        def get_type_to_actions_dicts(self):
            return grammar.get_type_to_actions_dict()

        @interface.implement
        def get_name_to_id_dicts(self):
            return grammar.get_name_to_id_dicts()

    if name is not None:
        BasicSearchState.__name__ = BasicSearchState.__qualname__ = name

    return BasicSearchState


class Compiler:
    @abstractfunction
    def compile_trees(self, trees):
        pass

    def compile_tree(self, tree):
        [executable] = self.compile_trees([tree])
        return executable
