import argparse
# import math

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval


KQAPRO_TRAN_SET_SIZE = 94376


def _parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_module', help='a config module (e.g. config.test_general)')
    parser.add_argument('--train-set-percent', dest='train_set_percent', type=float, help='train set percent to be used')

    args, unknown = parser.parse_known_args()
    return args


def _get_train_set_percent():
    args = _parse_cmd_args()
    return args.train_set_percent


def _extract_dataset_portion(dataset, percent, expected_dataset_size=None):
    # ratio = percent / 100
    num_examples = round(len(dataset) * percent / 100)
    if expected_dataset_size is not None:
        assert num_examples == expected_dataset_size
    return dataset[:num_examples]


def make_config(percent, shuffled_encoded_train_set_name):
    assert 0 < percent <= 100

    num_epoch_repeats = 100 / percent
    # num_epoch_repeats = math.sqrt(100 / percent)
    num_used_train_examples = round(KQAPRO_TRAN_SET_SIZE * percent / 100)

    if percent == 100:
        encoded_train_set = LazyEval(lambda: configuration.config.__getattr__(shuffled_encoded_train_set_name))
    else:
        encoded_train_set = LazyEval(lambda: _extract_dataset_portion(
            configuration.config.__getattr__(shuffled_encoded_train_set_name),
            percent=percent,
            expected_dataset_size=num_used_train_examples))

    return Environment(
        num_epoch_repeats=num_epoch_repeats,
        num_used_train_examples=num_used_train_examples,
        encoded_train_set=encoded_train_set,
    )


def make_config_from_cmd(shuffled_encoded_train_set_name='shuffled_encoded_train_set'):
    percent = _get_train_set_percent()
    return make_config(percent, shuffled_encoded_train_set_name)


config = make_config_from_cmd()
