
import os

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.filesys import get_new_path_with_number

# from utility.time import initial_date_str


_test_dir_root_path = './model-test'


def get_test_dir_path():
    model_learning_dir_name = os.path.basename(configuration.config.model_learning_dir_path)
    dir_name_sep = '#'
    common_test_dir_path = os.path.join(_test_dir_root_path, model_learning_dir_name + dir_name_sep)

    new_test_dir_path = get_new_path_with_number(common_test_dir_path)
    return new_test_dir_path


config = Environment(
    # run_mode='test',
    test_dir_path=LazyEval(get_test_dir_path),
    test_batch_size=64,
    # test_batch_size=1,
)
