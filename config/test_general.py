
import os
import argparse

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.filesys import get_new_path_with_number

# from utility.time import initial_date_str


_test_dir_root_path = './model-test'


def get_test_dir_path():
    if 'model_learning_dir_path' in configuration.config:
        test_dir_name = os.path.basename(configuration.config.model_learning_dir_path)
    else:
        assert 'model_checkpoint_dir_path' in configuration.config
        parser = argparse.ArgumentParser(description='Testing a semantic parser')
        parser.add_argument('--test-dir-name', dest='test_dir_name', help='a name of the directory for test output')
        args, unknown = parser.parse_known_args()
        test_dir_name = args.test_dir_name
        assert test_dir_name is not None

    dir_name_sep = '#'
    common_test_dir_path = os.path.join(_test_dir_root_path, test_dir_name + dir_name_sep)

    new_test_dir_path = get_new_path_with_number(common_test_dir_path)
    os.makedirs(new_test_dir_path)

    return new_test_dir_path


config = Environment(
    # run_mode='test',
    test_dir_path=LazyEval(get_test_dir_path),
    test_batch_size=64,
    # test_batch_size=1,
)
