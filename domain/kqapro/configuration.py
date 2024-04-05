
from dhnamlib.pylib.context import Environment, LazyEval, suppress_stdout, suppress_stderr, contextless, context_nest
from dhnamlib.pylib.filesys import json_load, jsonl_load

from splogic.base.grammar import read_grammar
from splogic.utility.acceleration import accelerator
from splogic.seq2seq import filemng
from splogic.base.execution import InstantExecutor
from splogic.seq2seq.dynamic_bind import UtteranceSpanTrieDynamicBinder, NoDynamicBinder

# from configuration import config as global_config
import configuration
# from configuration import coc

from .execution import (
    KQAProContext, KQAProDebugContext, KQAProCountingContext,
    KQAProExecResult, KQAProStricExectResult)
from .validation import KQAProDenotationEqual


_kb_file_path = './dataset/kqapro/kb.json'
_train_set_file_path = './dataset/kqapro/train.json'
_val_set_file_path = './dataset/kqapro/val.json'
_test_set_file_path = './dataset/kqapro/test.json'

_augmented_train_set_file_path = './processed/kqapro/augmented_train.jsonl'
_augmented_val_set_file_path = './processed/kqapro/augmented_val.jsonl'
_encoded_train_set_file_path = './processed/kqapro/encoded_train.jsonl'
_encoded_val_set_file_path = './processed/kqapro/encoded_val.jsonl'
_encoded_test_set_file_path = './processed/kqapro/encoded_test.jsonl'
_shuffled_augmented_train_set_file_path = './processed/kqapro/shuffled_augmented_train.jsonl'
_shuffled_encoded_train_set_file_path = './processed/kqapro/shuffled_encoded_train.jsonl'
# _encoded_train_mask_dataset_file_path = './processed/kqapro/encoded_train_mask.jsonl'
_augmented_strict_train_set_file_path = './processed/kqapro/augmented_strict_train.jsonl'
_augmented_strict_val_set_file_path = './processed/kqapro/augmented_strict_val.jsonl'
_encoded_strict_train_set_file_path = './processed/kqapro/encoded_strict_train.jsonl'
_encoded_strict_val_set_file_path = './processed/kqapro/encoded_strict_val.jsonl'
_shuffled_augmented_strict_train_set_file_path = './processed/kqapro/shuffled_augmented_strict_train.jsonl'
_shuffled_encoded_strict_train_set_file_path = './processed/kqapro/shuffled_encoded_strict_train.jsonl'
_encoded_weaksup_pretraining_set_file_path = './processed/kqapro/encoded_weaksup_pretraining.jsonl'
_encoded_weaksup_search_set_file_path = './processed/kqapro/encoded_weaksup_search.jsonl'

# _KQAPRO_TRAN_SET_SIZE = 94376
_USING_SPANS_AS_ENTITIES = False


_grammar_file_path = './domain/kqapro/grammar.lissp'
# _pretrained_model_name_or_path = './pretrained/bart-base'
_pretrained_model_name_or_path = 'facebook/bart-base'


def _make_context():
    with (
            contextless() if (configuration.config.using_tqdm and accelerator.is_local_main_process) else
            context_nest(suppress_stdout(), suppress_stderr())
    ):
        _context_cls = KQAProDebugContext if configuration.config.debug else KQAProContext
        return _context_cls(configuration.config.kb)

def _make_grammar():
    from .grammar import KQAProGrammar
    return read_grammar(
        _grammar_file_path,
        grammar_cls=KQAProGrammar,
        grammar_kwargs=dict(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path
        ))


ALL_RUN_MODES = [
    'train-default', 'train-for-multiple-decoding-strategies', 'test-on-val-set', 'test-on-test-set',
    'oracle-test-on-val-set', 'search-train'
    # 'retrain', 'finetune',
]

TRAINING_RUN_MODES = [
    'train-default', 'train-for-multiple-decoding-strategies', 'search-train'
]


config = Environment(
    # register=Register(strategy='conditional'),
    # domain='kqapro',

    kb=LazyEval(lambda: json_load(_kb_file_path)),
    # context=LazyEval(lambda: _context_cls(apply_recursively(config.kb))),
    # context=LazyEval(lambda: _context_cls(config.kb)),
    context=LazyEval(_make_context),
    test_executor=InstantExecutor(result_cls=KQAProExecResult, context_wrapper=KQAProCountingContext),
    search_executor=InstantExecutor(result_cls=KQAProStricExectResult, context_wrapper=KQAProCountingContext),
    denotation_equal=KQAProDenotationEqual(),
    dynamic_binder=UtteranceSpanTrieDynamicBinder() if _USING_SPANS_AS_ENTITIES else NoDynamicBinder(),
    max_num_program_iterations=200000,
    optim_measures=filemng.optim_measures,
    search_measures=filemng.search_measures,

    raw_train_set=LazyEval(lambda: json_load(_train_set_file_path)),
    raw_val_set=LazyEval(lambda: json_load(_val_set_file_path)),
    raw_test_set=LazyEval(lambda: json_load(_test_set_file_path)),

    augmented_train_set=LazyEval(lambda: jsonl_load(_augmented_train_set_file_path)),
    augmented_val_set=LazyEval(lambda: jsonl_load(_augmented_val_set_file_path)),
    encoded_train_set=LazyEval(lambda: jsonl_load(_encoded_train_set_file_path)),
    encoded_val_set=LazyEval(lambda: jsonl_load(_encoded_val_set_file_path)),
    encoded_test_set=LazyEval(lambda: jsonl_load(_encoded_test_set_file_path)),
    # encoded_train_mask_dataset=LazyEval(lambda: pickle_load(_encoded_train_mask_dataset_file_path)),
    shuffled_augmented_train_set=LazyEval(lambda: jsonl_load(_shuffled_augmented_train_set_file_path)),
    shuffled_encoded_train_set=LazyEval(lambda: jsonl_load(_shuffled_encoded_train_set_file_path)),
    augmented_strict_train_set=LazyEval(lambda: jsonl_load(_augmented_strict_train_set_file_path)),
    augmented_strict_val_set=LazyEval(lambda: jsonl_load(_augmented_strict_val_set_file_path)),
    encoded_strict_train_set=LazyEval(lambda: jsonl_load(_encoded_strict_train_set_file_path)),
    encoded_strict_val_set=LazyEval(lambda: jsonl_load(_encoded_strict_val_set_file_path)),
    shuffled_augmented_strict_train_set=LazyEval(lambda: jsonl_load(_shuffled_augmented_strict_train_set_file_path)),
    shuffled_encoded_strict_train_set=LazyEval(lambda: jsonl_load(_shuffled_encoded_strict_train_set_file_path)),
    encoded_weaksup_pretraining_set=LazyEval(lambda: jsonl_load(_encoded_weaksup_pretraining_set_file_path)),
    encoded_weaksup_search_set=LazyEval(lambda: jsonl_load(_encoded_weaksup_search_set_file_path)),

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
    shuffled_augmented_train_set_file_path=_shuffled_augmented_train_set_file_path,
    shuffled_encoded_train_set_file_path=_shuffled_encoded_train_set_file_path,
    augmented_strict_train_set_file_path=_augmented_strict_train_set_file_path,
    augmented_strict_val_set_file_path=_augmented_strict_val_set_file_path,
    encoded_strict_train_set_file_path=_encoded_strict_train_set_file_path,
    encoded_strict_val_set_file_path=_encoded_strict_val_set_file_path,
    shuffled_augmented_strict_train_set_file_path=_shuffled_augmented_strict_train_set_file_path,
    shuffled_encoded_strict_train_set_file_path=_shuffled_encoded_strict_train_set_file_path,
    encoded_weaksup_pretraining_set_file_path=_encoded_weaksup_pretraining_set_file_path,
    encoded_weaksup_search_set_file_path=_encoded_weaksup_search_set_file_path,

    # generation_max_length=500,
    generation_max_length=200,
    num_prediction_beams=1,
    # num_prediction_beams=4,
    softmax_masking=False,
    constrained_decoding=True,
    inferencing_subtypes=True,
    using_distinctive_union_types=True,

    # # ignoring_parsing_errors=False,
    # ignoring_parsing_errors=True,
)
