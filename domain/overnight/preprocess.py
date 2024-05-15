
import os
from argparse import ArgumentParser
# from itertools import chain
from tqdm import tqdm
# import torch
# import random

from meta_configuration import set_default_domain_name
set_default_domain_name('overnight')  # Call `set_default_domain_name` before the `configuration` module is loaded.
from configuration import config

from dhnamlib.pylib.iteration import rmap, flatten
from dhnamlib.pylib.filesys import jsonl_save, mkpdirs_unless_exist, pandas_tsv_load
# from dhnamlib.pylib.filesys import jsonl_save, pickle_save, mkpdirs_unless_exist, pandas_tsv_load
# from dhnamlib.pylib.time import TimeMeasure
# from dhnamlib.pylib.decoration import construct

# from splogic.seq2seq import learning
# from splogic.seq2seq.dynamic_bind import UtteranceSpanTrieDynamicBinder
# from splogic.base.formalism import InvalidCandidateActionError

# from .configuration import _make_grammar
from .grammar import DomainDynamicBinder

# from .lf_interface.transfer import labeled_logical_form_to_action_name_tree


@config
def extract_action_trees(domain, raw_dataset, grammar=config.ph, verifying=True, verifying_grammar=True):
    # properties of raw_dataset: utterance, logical_form, original

    action_trees = []
    for idx, row in raw_dataset.iterrows():
        action_tree = grammar.token_processing.labeled_logical_form_to_action_tree(
            row['logical_form'], grammar=grammar, domain=domain)
        action_trees.append(action_tree)

    if verifying_grammar:
        dynamic_binder = DomainDynamicBinder()
        compiler = grammar.compiler_cls()

        for idx, action_tree in enumerate(tqdm(action_trees)):
            action_seq = flatten(action_tree)[1:]  # except the starting action "program"
            dynamic_binding = dynamic_binder.bind(grammar, dict(domain=domain))
            with grammar.dynamic_scope.let(**dynamic_binding):
                last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=verifying)
            program = compiler.compile_tree(last_state.tree)
            if raw_dataset[idx]['logical_form'] != program:
                raise Exception('Original: {} | Composed: {}', raw_dataset[idx]['logical_form'], program)

    return action_trees

def augment_dataset(domain, raw_dataset, adding_action_name_tree=False, adding_answer_by_program=False):
    if adding_action_name_tree:
        print('Extracting action sequences from a dataset')
        action_trees = extract_action_trees(domain, raw_dataset)
        assert len(raw_dataset) == len(action_trees)
    print('Augmenting the dataset')
    augmented_dataset = [dict(utterance=row['utterance']) for idx, row in raw_dataset.iterrows()]
    for example_idx, example in enumerate(tqdm(augmented_dataset)):
        example['example_id'] = example_idx
        if adding_action_name_tree:
            action_tree = action_trees[example_idx]
            action_name_tree = rmap(lambda action: action.name, action_tree)
            example['action_name_tree'] = action_name_tree
        if adding_answer_by_program:
            raise NotImplementedError
    return augmented_dataset


@config
def preprocess_for_augmented_dataset(
        *,
        raw_dataset_dir_path,
        augmented_dataset_dir_path,
        adding_action_name_tree,
        # adding_answer_by_program,
        all_domains=config.ph,
        dataset_split,
):

    for domain in all_domains:
        raw_dataset_file_path = os.path.join(raw_dataset_dir_path, f'{domain}_{dataset_split}.tsv')
        augmented_dataset_file_path = os.path.join(augmented_dataset_dir_path, f'{domain}.jsonl')

        raw_dataset = pandas_tsv_load(raw_dataset_file_path, containing_header=True)

        augmented_dataset = augment_dataset(
            domain=domain,
            raw_dataset=raw_dataset,
            adding_action_name_tree=adding_action_name_tree,
            # adding_answer_by_program=adding_answer_by_program,
        )
        mkpdirs_unless_exist(augmented_dataset_file_path)
        jsonl_save(augmented_dataset, augmented_dataset_file_path)
        print(f'The augmented dataset was saved as {augmented_dataset_file_path}')


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
        ])

    args = parser.parse_args(config.remaining_cmd_args)

    if args.goal == 'augmented_train_set':
        preprocess_for_augmented_dataset(
            raw_dataset_dir_path=config.raw_dataset_dir_path,
            augmented_dataset_dir_path=config.augmented_dataset_dir_path,
            adding_action_name_tree=True,
            # adding_answer_by_program=False,
            dataset_split='train'
        )
    # elif args.goal == 'augmented_val_set':
    #     preprocess_for_augmented_dataset(
    #         raw_dataset=config.raw_val_set,
    #         augmented_dataset_file_path=config.augmented_val_set_file_path,
    #         adding_action_name_tree=True,
    #         adding_answer_by_program=False)
    # elif args.goal == 'encoded_train_set':
    #     preprocess_for_encoded_dataset(
    #         augmented_dataset=config.augmented_train_set,
    #         encoded_dataset_file_path=config.encoded_train_set_file_path)
    # elif args.goal == 'encoded_val_set':
    #     preprocess_for_encoded_dataset(
    #         augmented_dataset=config.augmented_val_set,
    #         encoded_dataset_file_path=config.encoded_val_set_file_path)
    # elif args.goal == 'encoded_test_set':
    #     preprocess_for_encoded_dataset(
    #         augmented_dataset=config.raw_test_set,
    #         encoded_dataset_file_path=config.encoded_test_set_file_path,
    #         example_idx_as_id=True)
    else:
        raise Exception('Unexpected goal')


if __name__ == '__main__':
    _main()
