
import os
import glob

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.decoration import variable, construct

# from utility.time import initial_date_str


_test_dir_root_path = './model-test'


def get_test_dir_path():
    model_learning_dir_name = os.path.basename(configuration.config.model_learning_dir_path)
    dir_name_sep = '#'
    common_test_dir_path = os.path.join(_test_dir_root_path, model_learning_dir_name)
    test_dir_paths = glob.glob(common_test_dir_path + dir_name_sep + '*')

    if len(test_dir_paths) > 0:
        @variable
        @construct(sorted)
        def test_numbers():
            for test_dir_path in test_dir_paths:
                _common_test_dir_path, number = test_dir_path.split(dir_name_sep)
                assert _common_test_dir_path == common_test_dir_path
                yield int(number)

        new_test_number = test_numbers[-1] + 1
    else:
        new_test_number = 0

    new_test_dir_path = common_test_dir_path + dir_name_sep + str(new_test_number)
    return new_test_dir_path


config = Environment(
    run_mode='test',
    test_dir_path=LazyEval(get_test_dir_path),
    test_batch_size=64,
)
