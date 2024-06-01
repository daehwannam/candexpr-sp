
import os
from functools import cache, partial

from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.debug import NIE
from dhnamlib.pylib.filesys import jsonl_load
from dhnamlib.pylib.decoration import construct

from splogic.base.grammar import read_grammar
# from splogic.seq2seq.validation import Validator, ResultCollector
from splogic.seq2seq.validation import NaiveDenotationEqual, Validator
from splogic.seq2seq import filemng
from splogic.base.execution import ExprCompiler

import configuration
from utility.exception import UVE

from .path import get_preprocessed_dataset_file_path
from .lf_interface.transfer import OVERNIGHT_DOMAINS
from .execution import OvernightContextCreater, OvernightExecutor
from .validation import OvernightResultCollector
# from .validation import OvernightDenotationEqual
from .filemng import save_analysis, save_extra_performance
from .dynamic_bind import DomainDynamicBinder
from splogic.seq2seq.data_read import make_data_loader


_pretrained_model_name_or_path = 'facebook/bart-base'

_raw_dataset_dir_path = './dataset/overnight'
_augmented_dataset_dir_path = './preprocessed/overnight/augmented'
_encoded_dataset_dir_path = './preprocessed/overnight/encoded'
_shuffled_encoded_dataset_dir_path = './preprocessed/overnight/shuffled_encoded'

_grammar_file_path = './domain/overnight/grammar.lissp'

_NO_CONTEXT = object()


def _make_grammar(**kwargs):
    from .grammar import OvernightGrammar
    return read_grammar(
        _grammar_file_path,
        grammar_cls=OvernightGrammar,
        grammar_kwargs=dict(
            pretrained_model_name_or_path=configuration.config.pretrained_model_name_or_path,
            **kwargs
        ))


_TRAIN_SET_RATIO = 0.8


@cache
def _load_train_val_sets():
    merged_train_set = []
    merged_val_set = []
    dataset_split = 'train'
    for domain in configuration.config.train_domains:
        dataset = jsonl_load(get_preprocessed_dataset_file_path(
            configuration.config.shuffled_encoded_dataset_dir_path, domain, dataset_split))
        _augment_dataset_with_domains(dataset, domain)

        train_set_size = int(len(dataset) * _TRAIN_SET_RATIO)

        train_set = dataset[:train_set_size]
        merged_train_set.extend(train_set)

        val_set = dataset[train_set_size:]
        _augment_dataset_with_answers(val_set, domain)
        merged_val_set.extend(val_set)

    return merged_train_set, merged_val_set


def _load_test_set():
    merged_test_set = []
    dataset_split = 'test'
    for domain in configuration.config.test_domains:
        dataset = jsonl_load(get_preprocessed_dataset_file_path(
            configuration.config.encoded_dataset_dir_path, domain, dataset_split))
        _augment_dataset_with_domains(dataset, domain)
        _augment_dataset_with_answers(dataset, domain)
        merged_test_set.extend(dataset)

    return merged_test_set


def _augment_dataset_with_domains(dataset, domain):
    for example in dataset:
        assert 'domain' not in example
        example['domain'] = domain


# @construct(list)
def _augment_dataset_with_answers(dataset, domain):
    logical_forms = [example['logical_form'] for example in dataset]
    executor = OvernightExecutor()
    contexts = (dict(domain=domain),) * len(logical_forms)
    exec_result = executor.execute(logical_forms, contexts)
    answers = exec_result.get()

    for example, answer in zip(dataset, answers):
        # augmented_example = dict(example)
        # augmented_example['answer'] = answer
        # yield augmented_example
        example['answer'] = answer


def _make_validator():
    return Validator(
        compiler=ExprCompiler(),
        context_creator=OvernightContextCreater(),
        executor=OvernightExecutor(),
        dynamic_binder=DomainDynamicBinder(),
        denotation_equal=NaiveDenotationEqual(),
        result_collector_cls=OvernightResultCollector,
        extra_analysis_keys=['domain'],
        evaluating_in_progress=False,
    )


config = Environment(
    optim_measures=filemng.optim_measures,
    # search_measures=filemng.search_measures,

    using_arg_candidate=True,
    using_arg_filter=False,
    constrained_decoding=True,
    inferencing_subtypes=True,
    using_distinctive_union_types=True,
    pretrained_model_name_or_path=_pretrained_model_name_or_path,
    # context=_NO_CONTEXT,
    grammar=LazyEval(_make_grammar),
    # compiler=LazyEval(NIE),
    test_validator=LazyEval(_make_validator),
    make_data_loader_fn=partial(make_data_loader, extra_keys=['domain']),
    save_analysis_fn=save_analysis,
    save_extra_performance_fn=save_extra_performance,

    # generation_max_length=500,
    generation_max_length=200,
    num_prediction_beams=1,
    # num_prediction_beams=4,
    softmax_masking=False,

    all_domains=OVERNIGHT_DOMAINS,
    # train_domains=LazyEval(UVE),
    # test_domains=LazyEval(UVE),
    # train_domains=OVERNIGHT_DOMAINS,
    # test_domains=OVERNIGHT_DOMAINS,

    raw_dataset_dir_path=_raw_dataset_dir_path,
    augmented_dataset_dir_path=_augmented_dataset_dir_path,
    encoded_dataset_dir_path=_encoded_dataset_dir_path,
    shuffled_encoded_dataset_dir_path=_shuffled_encoded_dataset_dir_path,

    encoded_train_set=LazyEval(lambda: _load_train_val_sets()[0]),
    encoded_val_set=LazyEval(lambda: _load_train_val_sets()[1]),
    encoded_test_set=LazyEval(_load_test_set),

    # context_creator=OvernightContextCreater(),
    # denotation_equal=OvernightDenotationEqual(),
    # result_collector_cls=OvernightResultCollector,
)
