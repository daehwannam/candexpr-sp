
import argparse
from functools import cache

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import not_none_valued_dict

import configuration

from ..common.train import get_model_learning_dir_path


_DOMAIN_NAME = 'overnight'
_MODEL_LEARNING_DIR_ROOT_PATH = f'./model-instance/{_DOMAIN_NAME}'


@cache
def parse_args():
    parser = argparse.ArgumentParser(description='Learning a semantic parser on the Overnight dataset')
    parser.add_argument('--unused-domain', dest='unused_domain', help='an unused domain during training')
    parser.add_argument('--single-domain', dest='single_domain', help='a domain used during training')
    args, unknown = parser.parse_known_args()

    arg_dict = not_none_valued_dict(vars(args))

    assert not (('single_domain' in arg_dict) and ('unused_domain' in arg_dict)),\
        '"--unused-domain" and "--single-domain" cannot be used together.'

    return arg_dict


def _get_train_domains():
    arg_dict = parse_args()
    if 'single_domain' in arg_dict:
        assert 'unused_domain' not in arg_dict
        train_domains = [arg_dict['single_domain']]
    elif 'unused_domain' in arg_dict:
        train_domains = list(configuration.config.all_domains)
        train_domains.remove(arg_dict['unused_domain'])
    else:
        train_domains = configuration.config.all_domains
    return train_domains


config = Environment(not_none_valued_dict(
    model_learning_dir_path=get_model_learning_dir_path(_MODEL_LEARNING_DIR_ROOT_PATH),
    train_domains=LazyEval(_get_train_domains),

    learning_rate=3e-5,
    # learning_rate=1e-3,
    adam_epsilon=1e-8,
    weight_decay=1e-5,
    # train_batch_size=16,
    train_batch_size=64,
    # val_batch_size=LazyEval(lambda: config.train_batch_size * 4),
    val_batch_size=LazyEval(lambda: config.train_batch_size),
    # val_batch_size=1,
    num_train_epochs=25,
    using_scheduler=True,
    num_warmup_epochs=LazyEval(lambda: config.num_train_epochs / 10),
    max_grad_norm=1,
    saving_optimizer=False,
    patience=float('inf'),
).items())
