
import os
from itertools import chain

from dhnamlib.pylib.context import Scope, LazyEval
from dhnamlib.pylib.decorators import Register
from dhnamlib.pylib.filesys import json_load, jsonl_load
from dhnamlib.pylib.iteration import apply_recursively, distinct_pairs
from dhnamlib.pylib.package import import_from_module

from logic.grammar import read_grammar

from domain.kqapro.execution import KoPLContext, KoPLDebugContext

_kb_file_path = './dataset/kqapro/kb.json'
_train_set_file_path = './dataset/kqapro/train.json'
_val_set_file_path = './dataset/kqapro/val.json'
_test_set_file_path = './dataset/kqapro/test.json'

_augmented_train_set_file_path = './processed/kqapro/augmented_train.jsonl'
_augmented_val_set_file_path = './processed/kqapro/augmented_val.jsonl'
_encoded_train_set_file_path = './processed/kqapro/encoded_train.jsonl'
_encoded_val_set_file_path = './processed/kqapro/encoded_val.jsonl'

_grammar_file_path = './domain/kqapro/grammar.lissp'
_pretrained_model_name_or_path = './pretrained/bart-base'

_DEBUG = True
if _DEBUG:
    _context_cls = KoPLContext
    _default_config_module_name = None
else:
    _context_cls = KoPLDebugContext
    _default_config_module_name = None

def make_grammar():
    from domain.kqapro.grammar import KoPLGrammar
    return read_grammar(_grammar_file_path, grammar_cls=KoPLGrammar)
    

_config = Scope(
    register=Register(strategy='conditional'),

    kb=LazyEval(lambda: json_load(_kb_file_path)),
    context=LazyEval(lambda: _context_cls(apply_recursively(config.kb))),

    raw_train_set=LazyEval(lambda: json_load(_train_set_file_path)),
    raw_val_set=LazyEval(lambda: json_load(_val_set_file_path)),
    raw_test_set=LazyEval(lambda: json_load(_test_set_file_path)),

    augmented_train_set=LazyEval(lambda: jsonl_load(_augmented_train_set_file_path)),
    augmented_val_set=LazyEval(lambda: jsonl_load(_augmented_val_set_file_path)),
    encoded_train_set=LazyEval(lambda: jsonl_load(_encoded_train_set_file_path)),
    encoded_val_set=LazyEval(lambda: jsonl_load(_encoded_val_set_file_path)),

    grammar=LazyEval(make_grammar),
    
    pretrained_model_name_or_path=_pretrained_model_name_or_path,

    augmented_train_set_file_path=_augmented_train_set_file_path,
    augmented_val_set_file_path=_augmented_val_set_file_path,
    encoded_train_set_file_path=_encoded_train_set_file_path,
    encoded_val_set_file_path=_encoded_val_set_file_path,
)

_specific_config_module_name = _default_config_module_name or os.environ.get('SEMPARSE_CONFIG')
if _specific_config_module_name is not None:
    _specific_config = import_from_module(_specific_config_module_name, 'config')
    config = Scope(distinct_pairs(chain(_config.items(), _specific_config.items())))
else:
    config = _config
