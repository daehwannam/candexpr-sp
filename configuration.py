
import os
from itertools import chain
from importlib import import_module
import argparse
import shutil

from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.decoration import Register, lru_cache, variable
from dhnamlib.pylib.filesys import json_load, json_save, jsonl_load, mkpdirs_unless_exist, mkloc_unless_exist, make_logger, get_new_path_with_number
# from dhnamlib.pylib.filesys import pickle_load
# from dhnamlib.pylib.iteration import apply_recursively
from dhnamlib.pylib.iteration import distinct_pairs, not_none_valued_pairs
from dhnamlib.pylib.text import parse_bool
# from dhnamlib.pylib.package import import_from_module
from dhnamlib.pylib.version_control import get_git_hash

from utility.time import initial_date_str
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
_encoded_test_set_file_path = './processed/kqapro/encoded_test.jsonl'
# _encoded_train_mask_dataset_file_path = './processed/kqapro/encoded_train_mask.jsonl'

_grammar_file_path = './domain/kqapro/grammar.lissp'
# _pretrained_model_name_or_path = './pretrained/bart-base'
_pretrained_model_name_or_path = 'facebook/bart-base'

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


def _is_valid_run_mode(run_mode):
    return run_mode in ['train', 'retrain', 'finetune', 'test-on-val-set', 'test-on-test-set']


def _is_training_run_mode(run_mode):
    assert _is_valid_run_mode(config.run_mode)
    return run_mode in ['train', 'retrain', 'finetune']

def _make_logger():
    if _is_training_run_mode(config.run_mode):
        log_file_path = os.path.join(config.model_learning_dir_path, f'{initial_date_str}_{config.run_mode}.log')
        mkpdirs_unless_exist(log_file_path)
    else:
        log_file_path = None
    return make_logger(
        name=config.run_mode,
        log_file_path=log_file_path)


def _get_xtqdm():
    if config.using_tqdm:
        from dhnamlib.pylib.iteration import xtqdm
    else:
        def xtqdm(iterator, /, *args, **kwargs):
            return iterator
    return xtqdm


_default_config = Environment(
    register=Register(strategy='conditional'),

    kb=LazyEval(lambda: json_load(_kb_file_path)),
    # context=LazyEval(lambda: _context_cls(apply_recursively(config.kb))),
    context=LazyEval(lambda: _context_cls(config.kb)),
    max_num_program_iterations=200000,

    raw_train_set=LazyEval(lambda: json_load(_train_set_file_path)),
    raw_val_set=LazyEval(lambda: json_load(_val_set_file_path)),
    raw_test_set=LazyEval(lambda: json_load(_test_set_file_path)),

    augmented_train_set=LazyEval(lambda: jsonl_load(_augmented_train_set_file_path)),
    augmented_val_set=LazyEval(lambda: jsonl_load(_augmented_val_set_file_path)),
    encoded_train_set=LazyEval(lambda: jsonl_load(_encoded_train_set_file_path)),
    encoded_val_set=LazyEval(lambda: jsonl_load(_encoded_val_set_file_path)),
    encoded_test_set=LazyEval(lambda: jsonl_load(_encoded_test_set_file_path)),
    # encoded_train_mask_dataset=LazyEval(lambda: pickle_load(_encoded_train_mask_dataset_file_path)),

    grammar=LazyEval(_make_grammar),
    compiler=LazyEval(lambda: config.grammar.compiler_cls()),
    using_arg_candidate=True,
    using_arg_filter=False,
    
    pretrained_model_name_or_path=_pretrained_model_name_or_path,

    augmented_train_set_file_path=_augmented_train_set_file_path,
    augmented_val_set_file_path=_augmented_val_set_file_path,
    encoded_train_set_file_path=_encoded_train_set_file_path,
    encoded_val_set_file_path=_encoded_val_set_file_path,
    encoded_test_set_file_path=_encoded_test_set_file_path,
    # encoded_train_mask_dataset_file_path=_encoded_train_mask_dataset_file_path,

    device=LazyEval(_get_device),
    logger=LazyEval(_make_logger),

    # generation_max_length=500,
    generation_max_length=200,
    num_prediction_beams=1,
    # num_prediction_beams=4,
    softmax_masking=True,
    constrained_decoding=True,

    git_hash=get_git_hash(),
    debug=_DEBUG,
    # using_tqdm=True,
    xtqdm=LazyEval(_get_xtqdm),
)


@lru_cache
def _parse_cmd_args():
    parser = argparse.ArgumentParser(description='Semantic parsing')
    parser.add_argument('--config', dest='config_module', help='a config module (e.g. config.test_general)')
    parser.add_argument('--additional-config', dest='additional_config_module', help='an additional config module which can overwrite other configurations')
    parser.add_argument('--model-learning-dir', dest='model_learning_dir_path', help='a path to the directory of a model')
    # parser.add_argument('--run_mode', dest='run_mode', help='an execution run_mode', choices=['train', 'test'])
    parser.add_argument('--using-tqdm', dest='using_tqdm', type=parse_bool, default=True, help='whether using tqdm')

    args, unknown = parser.parse_known_args()
    cmd_arg_dict = dict(not_none_valued_pairs(vars(args).items()))
    cmd_arg_dict['remaining_cmd_args'] = unknown
    return cmd_arg_dict


@lru_cache
def _get_specific_config_module():
    cmd_arg_dict = _parse_cmd_args()
    if cmd_arg_dict.get('config_module') is not None:
        specific_config_module = import_module(cmd_arg_dict['config_module'])
    else:
        specific_config_module = None
    return specific_config_module


@lru_cache
def _get_additional_config_module():
    cmd_arg_dict = _parse_cmd_args()
    if cmd_arg_dict.get('additional_config_module') is not None:
        _additional_config_module = import_module(cmd_arg_dict['additional_config_module'])
    else:
        _additional_config_module = None
    return _additional_config_module


@variable
def config():
    cmd_arg_dict = _parse_cmd_args()
    specific_config_module = _get_specific_config_module()
    specific_config = (dict() if specific_config_module is None else
                       specific_config_module.config)
    additional_config_module = _get_additional_config_module()
    additional_config = (dict() if additional_config_module is None else
                         additional_config_module.config)
 
    config = Environment(
        chain(
            distinct_pairs(chain(
                _default_config.items(),
                specific_config.items(),
                cmd_arg_dict.items()
            )),
            additional_config.items()
        ))
    return config


@variable
def config_path_dict():
    specific_config_module = _get_specific_config_module()
    specific_config_module_file_path = (None if specific_config_module is None else
                                        specific_config_module.__file__)
    additional_config_module = _get_additional_config_module()
    additional_config_module_file_path = (None if additional_config_module is None else
                                          additional_config_module.__file__)
    return dict(
        general=__file__,
        specific=specific_config_module_file_path,
        additional=additional_config_module_file_path
    )


def _config_to_json_dict(config):
    return dict([key, value] for key, value in config.items()
                if not isinstance(value, (LazyEval, Register)))


def save_config_info(dir_path):
    json_dict = _config_to_json_dict(config)
    config_info_path = get_new_path_with_number(
        os.path.join(dir_path, f'config-info-{config.run_mode}'),
        starting_num=1, no_first_num=True
    )
    mkloc_unless_exist(config_info_path)
    json_save(json_dict, os.path.join(config_info_path, 'config.json'))
    json_save(config_path_dict, os.path.join(config_info_path, 'config-path.json'))
    shutil.copytree('./config', os.path.join(config_info_path, 'config'))
    shutil.copyfile('./configuration.py', os.path.join(config_info_path, 'configuration.py'))


# if 'run_mode' in config and _is_training_run_mode(config.run_mode):
#     save_config_info(config.model_learning_dir_path)
