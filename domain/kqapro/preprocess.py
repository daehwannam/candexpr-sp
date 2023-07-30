
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
import torch
import random

from configuration import config, _make_grammar

from dhnamlib.pylib.filesys import jsonl_save, pickle_save, mkpdirs_unless_exist
from dhnamlib.pylib.time import TimeMeasure
from dhnamlib.pylib.iteration import apply_recursively
from dhnamlib.pylib.decoration import construct

from .execution import postprocess_prediction
from .kopl_original import execute_kopl_program
from . import kopl_transfer
from . import learning


@config
def extract_action_seqs(raw_dataset, grammar=config.ph, context=config.ph, verbose=1, verifying=True, verifying_grammar=True):
    compiler = grammar.compiler_cls()

    tm = TimeMeasure()

    kopl_to_action_seq_cumtime = 0
    get_last_state_cumtime = 0
    compile_tree_cumtime = 0
    program_cumtime = 0
    postprocess_prediction_cumtime = 0

    action_seqs = []

    with TimeMeasure() as total_tm:
        for example_idx, example in tqdm(enumerate(raw_dataset), total=len(raw_dataset)):
            labeled_kopl_program = example['program']
            answer = example['answer']

            with tm:
                action_seq = kopl_transfer.kopl_to_action_seq(grammar, labeled_kopl_program)
            kopl_to_action_seq_cumtime += tm.interval

            action_seqs.append(action_seq)

            if not verifying:
                continue

            with tm:
                if verifying_grammar:
                    utterance_token_id_seq = grammar.utterance_tokenizer(example['question'])['input_ids']
                    dynamic_trie = learning._utterance_token_id_seq_to_dynamic_trie(grammar, utterance_token_id_seq)
                    with grammar.let_dynamic_trie(dynamic_trie):
                        last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=verifying)
                else:
                    last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=verifying)
            get_last_state_cumtime += tm.interval

            with tm:
                program = compiler.compile_tree(last_state.tree)
            compile_tree_cumtime += tm.interval

            with tm:
                denotation = program(config.context)
            program_cumtime += tm.interval

            with tm:
                prediction = postprocess_prediction(denotation)
            postprocess_prediction_cumtime += tm.interval

            if verifying:
                denotation_by_kopl = execute_kopl_program(config.context, labeled_kopl_program)
                assert denotation == denotation_by_kopl

            if False and verbose >= 2:
                if answer != prediction:
                    denotation_by_kopl = execute_kopl_program(config.context, labeled_kopl_program)
                    if denotation == denotation_by_kopl:
                        prediction_by_kopl = postprocess_prediction(denotation_by_kopl)
                        assert prediction == prediction_by_kopl
                        print(f'The labeled answer of id {example_idx} is incorrect. Expected {prediction_by_kopl} but got {answer}')
                    else:
                        breakpoint()
                        print(f'Incorrect prediction for an example of index {example_idx}')

    cum_time_dict = dict(
        kopl_to_action_seq = kopl_to_action_seq_cumtime,
        get_last_state     = get_last_state_cumtime,
        compile_tree       = compile_tree_cumtime,
        program            = program_cumtime,
        postprocess_prediction = postprocess_prediction_cumtime)
    avg_time_dict = dict([k, v / len(raw_dataset)] for k, v in cum_time_dict.items())

    if verbose >= 1:
        print('Total time: {total_tm.interval}')
        print('=== Average time ===')
        for k, v in avg_time_dict.items():
            if v > 0:
                print(f'- {k}: {v}')

    return action_seqs


def action_seq_to_name_seq(action_seq):
    return [action.name for action in action_seq]


def name_seq_to_action_seq(grammar, action_name_seq):
    return list(map(grammar.name_to_action, action_name_seq))


def augment_dataset(raw_dataset, adding_action_name_seq=False, adding_answer_by_program=False, context=None):
    if adding_action_name_seq:
        print('Extracting action sequences from a dataset')
        action_seqs = extract_action_seqs(raw_dataset, verbose=0, verifying=False)
        assert len(raw_dataset) == len(action_seqs)
    print('Augmenting the dataset')
    augmented_dataset = apply_recursively(raw_dataset)
    for example_idx, example in tqdm(enumerate(augmented_dataset), total=len(augmented_dataset)):
        example['example_id'] = example_idx
        if adding_action_name_seq:
            action_seq = action_seqs[example_idx]
            action_name_seq = [action.name for action in action_seq]
            example['action_name_seq'] = action_name_seq
        if adding_answer_by_program:
            assert context is not None
            answer_by_program = postprocess_prediction(execute_kopl_program(context, example['program']))
            example['answer_by_program'] = answer_by_program
    return augmented_dataset


@config
def preprocess_for_augmented_dataset(
        *,
        raw_dataset,
        augmented_dataset_file_path,
        adding_action_name_seq,
        adding_answer_by_program,
        context=config.ph):

    augmented_dataset = augment_dataset(
        raw_dataset=raw_dataset,
        adding_action_name_seq=adding_action_name_seq,
        adding_answer_by_program=adding_answer_by_program,
        context=context)
    mkpdirs_unless_exist(augmented_dataset_file_path)
    jsonl_save(augmented_dataset, augmented_dataset_file_path)
    print(f'The augmented dataset was saved as {augmented_dataset_file_path}')


@construct(list)
def encode_dataset(grammar, augmented_dataset, example_idx_as_id=False):
    for example_idx, example in enumerate(tqdm(augmented_dataset)):

        # `utterance_token_ids` and `action_ids` include BOS and EOS.
        encoded_example = dict(
            example_id=example_idx if example_idx_as_id else example['example_id'],
            utterance_token_ids=grammar.utterance_tokenizer(example['question'])['input_ids'])

        if 'answer' in example:
            encoded_example.update(answer=example['answer'])

        if 'action_name_seq' in example:
            action_ids = list(chain(
                [grammar.lf_tokenizer.bos_token_id],
                map(grammar.name_to_id, example['action_name_seq']),
                [grammar.lf_tokenizer.eos_token_id]))
            encoded_example.update(action_ids=action_ids)

        if 'answer_by_program' in example:
            encoded_example.update(answer_by_program=example['answer_by_program'])

        yield encoded_example


@config
def preprocess_for_encoded_dataset(
        *,
        grammar=config.ph,
        augmented_dataset,
        encoded_dataset_file_path,
        example_idx_as_id=False):
    encoded_dataset = encode_dataset(grammar, augmented_dataset, example_idx_as_id=example_idx_as_id)
    mkpdirs_unless_exist(encoded_dataset_file_path)
    jsonl_save(encoded_dataset, encoded_dataset_file_path)
    print(f'The augmented dataset was saved as {encoded_dataset_file_path}')


@construct(list)
def encode_mask(grammar, encoded_dataset):
    for example in tqdm(encoded_dataset):
        labels = torch.tensor([example['action_ids'][1:]], dtype=torch.int64)
        softmax_mask, nll_mask = learning.labels_to_masks(grammar, labels)

        softmax_mask = tuple(map(bytes, softmax_mask[0].tolist()))
        nll_mask = bytes(nll_mask[0].tolist())

        yield dict(softmax_mask=softmax_mask,
                   nll_mask=nll_mask)


@config
def preprocess_for_encoded_mask(
        *,
        grammar=config.ph,
        encoded_dataset,
        encoded_train_mask_dataset_file_path):
    encoded_mask = encode_mask(grammar, encoded_dataset)
    mkpdirs_unless_exist(encoded_train_mask_dataset_file_path)
    pickle_save(encoded_mask, encoded_train_mask_dataset_file_path)
    print(f'The encoded mask was saved as {encoded_train_mask_dataset_file_path}')


def shuffle_dataset(dataset):
    seed = 42
    new_dataset = list(dataset)
    random.Random(seed).shuffle(new_dataset)
    return new_dataset


def preprocess_for_shuffled_dataset(
        *,
        dataset,
        shuffled_dataset_file_path):
    shuffled_dataset = shuffle_dataset(dataset)
    mkpdirs_unless_exist(shuffled_dataset_file_path)
    jsonl_save(shuffled_dataset, shuffled_dataset_file_path)
    print(f'The shuffled dataset was saved as {shuffled_dataset_file_path}')


@construct(list)
def augment_dataset_with_strict_grammar(augmented_dataset, grammar):
    assert grammar.inferencing_subtypes is False
    for example in tqdm(augmented_dataset):
        new_example = dict(example)
        action_seq = kopl_transfer.get_strictly_typed_action_seq(grammar, example['action_name_seq'])
        action_name_seq = [action.name for action in action_seq]
        new_example['action_name_seq'] = action_name_seq  # update 'action_name_seq'
        yield new_example


def preprocess_for_augmented_strict_dataset(
        *,
        augmented_dataset,
        augmented_strict_dataset_file_path):
    with config.let(inferencing_subtypes=False):
        grammar = _make_grammar()
        assert grammar.inferencing_subtypes is False

    augmented_strict_dataset = augment_dataset_with_strict_grammar(augmented_dataset, grammar)
    jsonl_save(augmented_strict_dataset, augmented_strict_dataset_file_path)
    print(f'The augmented dataset with strict grammar was saved as {augmented_strict_dataset_file_path}')


def process_for_encoded_strict_dataset(
        *,
        augmented_dataset,
        encoded_dataset_file_path,
        example_idx_as_id=False
):
    with config.let(inferencing_subtypes=False):
        grammar = _make_grammar()
        assert grammar.inferencing_subtypes is False

    preprocess_for_encoded_dataset(
        grammar=grammar,
        augmented_dataset=augmented_dataset,
        encoded_dataset_file_path=encoded_dataset_file_path,
        example_idx_as_id=example_idx_as_id)


def _main():
    parser = ArgumentParser(description='Preprocess KoPL dataset',)
    parser.add_argument(
        '--goal',
        choices=[
            'augmented_train_set',
            'augmented_val_set',
            'encoded_train_set',
            'encoded_val_set',
            'encoded_test_set',
            # 'encoded_train_mask',
            'shuffled_augmented_train_set',
            'shuffled_encoded_train_set',
            'augmented_strict_train_set',
            'augmented_strict_val_set',
            'encoded_strict_train_set',
            'encoded_strict_val_set',
            'shuffled_augmented_strict_train_set',
            'shuffled_encoded_strict_train_set',
        ])

    args = parser.parse_args(config.remaining_cmd_args)

    if args.goal == 'augmented_train_set':
        preprocess_for_augmented_dataset(
            raw_dataset=config.raw_train_set,
            augmented_dataset_file_path=config.augmented_train_set_file_path,
            adding_action_name_seq=True,
            adding_answer_by_program=True)
    elif args.goal == 'augmented_val_set':
        preprocess_for_augmented_dataset(
            raw_dataset=config.raw_val_set,
            augmented_dataset_file_path=config.augmented_val_set_file_path,
            adding_action_name_seq=True,
            adding_answer_by_program=True)
    elif args.goal == 'encoded_train_set':
        preprocess_for_encoded_dataset(
            augmented_dataset=config.augmented_train_set,
            encoded_dataset_file_path=config.encoded_train_set_file_path)
    elif args.goal == 'encoded_val_set':
        preprocess_for_encoded_dataset(
            augmented_dataset=config.augmented_val_set,
            encoded_dataset_file_path=config.encoded_val_set_file_path)
    elif args.goal == 'encoded_test_set':
        preprocess_for_encoded_dataset(
            augmented_dataset=config.raw_test_set,
            encoded_dataset_file_path=config.encoded_test_set_file_path,
            example_idx_as_id=True)
    elif args.goal == 'encoded_train_mask':
        raise NotImplementedError
        preprocess_for_encoded_mask(
            encoded_dataset=config.encoded_train_set,
            encoded_train_mask_dataset_file_path=config.encoded_train_mask_dataset_file_path)
    elif args.goal == 'shuffled_augmented_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.augmented_train_set,
            shuffled_dataset_file_path=config.shuffled_augmented_train_set_file_path)
    elif args.goal == 'shuffled_encoded_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.encoded_train_set,
            shuffled_dataset_file_path=config.shuffled_encoded_train_set_file_path)
    elif args.goal == 'augmented_strict_train_set':
        preprocess_for_augmented_strict_dataset(
            augmented_dataset=config.augmented_train_set,
            augmented_strict_dataset_file_path=config.augmented_strict_train_set_file_path)
    elif args.goal == 'augmented_strict_val_set':
        preprocess_for_augmented_strict_dataset(
            augmented_dataset=config.augmented_val_set,
            augmented_strict_dataset_file_path=config.augmented_strict_val_set_file_path)
    elif args.goal == 'encoded_strict_train_set':
        process_for_encoded_strict_dataset(
            augmented_dataset=config.augmented_strict_train_set,
            encoded_dataset_file_path=config.encoded_strict_train_set_file_path)
    elif args.goal == 'encoded_strict_val_set':
        process_for_encoded_strict_dataset(
            augmented_dataset=config.augmented_strict_val_set,
            encoded_dataset_file_path=config.encoded_strict_val_set_file_path)
    elif args.goal == 'shuffled_augmented_strict_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.augmented_strict_train_set,
            shuffled_dataset_file_path=config.shuffled_augmented_strict_train_set_file_path)
    elif args.goal == 'shuffled_encoded_strict_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.encoded_strict_train_set,
            shuffled_dataset_file_path=config.shuffled_encoded_strict_train_set_file_path)
    else:
        raise Exception('Unexpected goal')


if __name__ == '__main__':
    _main()
