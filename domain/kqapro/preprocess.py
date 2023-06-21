from tqdm import tqdm
from argparse import ArgumentParser
from itertools import chain

from configuration import config

from dhnamlib.pylib.filesys import jsonl_save, mkpdirs_unless_exist
from dhnamlib.pylib.time import TimeMeasure
from dhnamlib.pylib.iteration import apply_recursively
from dhnamlib.pylib.decoration import construct

from .execution import postprocess_prediction
from .kopl_original import execute_kopl_program
from . import kopl_transfer


@config
def extract_action_seqs(raw_dataset, grammar=config.ph, context=config.ph, verbose=1, verifying=True):
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
                last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=True)
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

            if verbose >= 2:
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
def encode_dataset(grammar, augmented_dataset):
    for example in tqdm(augmented_dataset):

        # `utterance_token_ids` and `action_ids` include BOS and EOS.
        encoded_example = dict(
            utterance_token_ids=grammar.utterance_tokenizer(example['question'])['input_ids'],
            answer=example['answer'])

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
        augmented_dataset=None,
        encoded_dataset_file_path):
    encoded_dataset = encode_dataset(grammar, augmented_dataset)
    mkpdirs_unless_exist(encoded_dataset_file_path)
    jsonl_save(encoded_dataset, encoded_dataset_file_path)
    print(f'The augmented dataset was saved as {encoded_dataset_file_path}')


def _main():
    parser = ArgumentParser(description='Preprocess KoPL dataset',)
    parser.add_argument(
        '--goal',
        choices=[
            'augmented_train_set',
            'augmented_val_set',
            'encoded_train_set',
            'encoded_val_set'
        ])

    args = parser.parse_args()

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
            adding_action_name_seq=False,
            adding_answer_by_program=True)
    elif args.goal == 'encoded_train_set':
        preprocess_for_encoded_dataset(
            augmented_dataset=config.augmented_train_set,
            encoded_dataset_file_path=config.encoded_train_set_file_path)
    elif args.goal == 'encoded_val_set':
        preprocess_for_encoded_dataset(
            augmented_dataset=config.augmented_val_set,
            encoded_dataset_file_path=config.encoded_val_set_file_path)
    else:
        raise Exception('Unexpected goal')


if __name__ == '__main__':
    _main()
