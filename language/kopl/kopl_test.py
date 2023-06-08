
from tqdm import tqdm

from dhnamlib.pylib.filesys import json_load, python_pretty_save

from .compile import postprocess_answer
from .kopl_original import execute_kopl_program


def test_dataset():
    from configuration import config

    dataset = json_load('./dataset/kopl/train.json')

    correct_count = 0
    incorrect_example_info = []

    for example_idx, example in tqdm(enumerate(dataset)):
        denotation_by_kopl = execute_kopl_program(config.context, example['program'])
        answer_by_kopl = postprocess_answer(denotation_by_kopl)
        if answer_by_kopl == example['answer']:
            correct_count += 1
        else:
            incorrect_example_info.append((example_idx, (example['answer'], answer_by_kopl)))

    print(f'#correct: {correct_count}')
    print(f'#incorrect: {len(dataset) - correct_count}')
    print(f'acc: {correct_count / len(dataset)}')

    incorrect_example_info_file_path = './_tmp_incorrect_example_info.txt'
    python_pretty_save(incorrect_example_info, incorrect_example_info_file_path)
    print(f'incorrect_example_info is saved as {incorrect_example_info_file_path}')


def test_dataset_size():
    dataset = json_load('./dataset/kopl/train.json')
    print(len(dataset))


def test_dataset_answers():
    dataset = json_load('./dataset/kopl/train.json')
    for example_idx, example in tqdm(enumerate(dataset)):
        if example['answer'] not in example['choices']:
            breakpoint
            print(example)
    print('done')


# context = KoPLEngine(json.load(open(os.path.join(args.input_dir, 'kb.json'))))
