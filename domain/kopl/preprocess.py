from tqdm import tqdm
from argparse import ArgumentParser

from configuration import config

from dhnamlib.pylib.filesys import jsonl_save, jsonl_load, mkdirs_unless_exist
from dhnamlib.pylib.time import TimeMeasure

from .execution import postprocess_answer
from .kopl_original import execute_kopl_program
from . import kopl_transfer


@config
def extract_action_seqs(dataset, grammar=config.ph, context=config.context, verbose=1, verifying=True):
    compiler = grammar.compiler_cls()

    tm = TimeMeasure()

    kopl_to_action_seq_cumtime = 0
    get_last_state_cumtime = 0
    compile_tree_cumtime = 0
    program_cumtime = 0
    postprocess_answer_cumtime = 0

    action_seqs = []

    with TimeMeasure() as total_tm:
        for example_idx, example in tqdm(enumerate(dataset)):
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
                prediction = postprocess_answer(denotation)
            postprocess_answer_cumtime += tm.interval

            if verbose >= 2:
                if answer != prediction:
                    denotation_by_kopl = execute_kopl_program(config.context, labeled_kopl_program)
                    if denotation == denotation_by_kopl:
                        prediction_by_kopl = postprocess_answer(denotation_by_kopl)
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
        postprocess_answer = postprocess_answer_cumtime)
    avg_time_dict = dict([k, v / len(dataset)] for k, v in cum_time_dict.items())

    if verbose >= 1:
        print('Total time: {total_tm.interval}')
        print('=== Average time ===')
        for k, v in avg_time_dict.items():
            if v > 0:
                print(f'- {k}: {v}')

    return action_seqs


def save_action_seqs(action_seqs, file_path):
    action_name_seqs = [[action.name for action in action_seq] for action_seq in action_seqs]
    mkdirs_unless_exist(file_path)
    jsonl_save(action_name_seqs, file_path)


def load_action_seqs(grammar, action_seqs_file_path):
    action_name_seqs = jsonl_load(action_seqs_file_path)
    action_seqs = tuple(map(grammar.name_to_action, action_name_seqs))
    return action_seqs


@config
def preprocess_action_seqs(
        dataset=config.ph.train_set,
        action_seqs_file_path=config.ph.train_action_seqs_file_path):

    action_seqs = extract_action_seqs(dataset, verbose=0, verifying=False)
    save_action_seqs(action_seqs, action_seqs_file_path)

    print(f'action sequences are saved in {action_seqs_file_path}')


def _main():
    parser = ArgumentParser(description='Preprocess KoPL dataset',)
    parser.add_argument('--datatype', choices=['action'], default='action')

    args = parser.parse_args()

    if args.datatype == 'action':
        preprocess_action_seqs()
    else:
        raise Exception('Unexpected datatype')


if __name__ == '__main__':
    _main()
