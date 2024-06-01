
import argparse
from functools import cache
from itertools import chain

from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.iteration import distinct_pairs

import configuration

from .train_general import config as _config_general


@cache
def parse_args():
    parser = argparse.ArgumentParser(description='Learning a semantic parser for multiple domains except a specific domain')
    parser.add_argument('--unused-domain', dest='unused_domain', help='an unused domain during training')
    args, unknown = parser.parse_known_args()

    return vars(args)


def _get_train_domains():
    arg_dict = parse_args()
    train_domains = list(configuration.config.all_domains)
    train_domains.remove(arg_dict['unused_domain'])
    return train_domains


_config_specific = Environment(
    run_mode='train-default',
    train_domains=LazyEval(_get_train_domains)
)

config = Environment(distinct_pairs(chain(
    _config_general.items(),
    _config_specific.items()
)))
