
import os
from itertools import chain
from importlib import import_module
import argparse
import datetime

from dhnamlib.pylib.context import Scope, LazyEval
from dhnamlib.pylib.decoration import Register, cache, variable
from dhnamlib.pylib.filesys import json_load, jsonl_load
from dhnamlib.pylib.iteration import apply_recursively, distinct_pairs
# from dhnamlib.pylib.package import import_from_module
from dhnamlib.pylib.filesys import make_logger

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

def _make_grammar():
    from domain.kqapro.grammar import KoPLGrammar
    return read_grammar(_grammar_file_path, grammar_cls=KoPLGrammar)


def _get_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


_current_datetime = datetime.datetime.now()

def _get_date_str():
    return _current_datetime.strftime("%Y-%m-%d_%H:%M:%S_%f")


def _make_logger():
    if config.mode in ['train', 'finetune']:
        log_file_path = os.path.join(config.model_dir_path, f'{_get_date_str()}_{config.mode}.log')
    else:
        log_file_path = None
    return make_logger(
        name=config.mode,
        log_file_path=log_file_path)


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

    grammar=LazyEval(_make_grammar),
    
    pretrained_model_name_or_path=_pretrained_model_name_or_path,

    augmented_train_set_file_path=_augmented_train_set_file_path,
    augmented_val_set_file_path=_augmented_val_set_file_path,
    encoded_train_set_file_path=_encoded_train_set_file_path,
    encoded_val_set_file_path=_encoded_val_set_file_path,

    device=LazyEval(_get_device),
    logger=LazyEval(_make_logger),

    batch_size=16,
)

@cache
def _parse_cmd_args():
    parser = argparse.ArgumentParser(description='Semantic parsing')
    parser.add_argument('--config', dest='config_module', help='a config module')
    parser.add_argument('--model-dir', dest='model_dir_path', help='a path to the directory of a moddel')

    args = parser.parse_args()
    return vars(args)


@cache
def _get_specific_config_module():
    _cmd_args = _parse_cmd_args()
    if _cmd_args['config_module'] is not None:
        _specific_config_module = import_module(_cmd_args['config_module'])
    else:
        _specific_config_module = None
    return _specific_config_module


@variable
def config():
    _cmd_args = _parse_cmd_args()
    _specific_config_module = _get_specific_config_module()
    _specific_config = (dict() if _specific_config_module is None else
                        _specific_config_module.config)
 
    _remaining_cmd_args = dict([k, v] for k, v in _cmd_args.items()
                               if ((k not in ['config_module']) and (v is not None)))

    config = Scope(
        distinct_pairs(chain(
            _config.items(),
            _specific_config.items(),
            _remaining_cmd_args.items()
        )))
    return config


@variable
def config_path_dict():
    _specific_config_module = _get_specific_config_module()
    _specific_config_module_file_path = (None if _specific_config_module is None else
                                         _specific_config_module.__file__)
    return dict(
        general=__file__,
        specific=_specific_config_module_file_path
    )
