import argparse
import math

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval


KQAPRO_TRAN_SET_SIZE = 94376


def _parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_module', help='a config module (e.g. config.test_general)')
    parser.add_argument('--train-set-percent', dest='train_set_percent', type=float, help='train set percent to be used')
    parser.add_argument('--num-epoch-repeats', dest='num_epoch_repeats', type=float, help='the number of epoch repeats before validation')
    parser.add_argument('--epoch-repeat-strategy', dest='epoch_repeat_strategy', type=str, choices=['linear', 'sqrt'], default='sqrt',
                        help='the way to compute the number of epoch repeats')

    args, unknown = parser.parse_known_args()
    return args


def _extract_dataset_portion(dataset, percent, expected_dataset_size=None):
    # ratio = percent / 100
    num_examples = round(len(dataset) * percent / 100)
    if expected_dataset_size is not None:
        assert num_examples == expected_dataset_size
    return dataset[:num_examples]


def make_config(percent, shuffled_encoded_train_set_name, num_epoch_repeats, epoch_repeat_strategy):
    assert 0 < percent <= 100

    if num_epoch_repeats is None:
        if epoch_repeat_strategy == 'linear':
            num_epoch_repeats = 100 / percent
        elif epoch_repeat_strategy == 'sqrt':
            num_epoch_repeats = math.sqrt(100 / percent)
        else:
            raise Exception('Unknown epoch_repeat_strategy')
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
    args = _parse_cmd_args()
    percent = args.train_set_percent
    num_epoch_repeats = args.num_epoch_repeats
    epoch_repeat_strategy = args.epoch_repeat_strategy
    return make_config(percent, shuffled_encoded_train_set_name, num_epoch_repeats, epoch_repeat_strategy)


config = make_config_from_cmd()
