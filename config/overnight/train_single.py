
import argparse
from functools import cache
from itertools import chain

from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.iteration import distinct_pairs

from .train_general import config as _config_general


@cache
def parse_args():
    parser = argparse.ArgumentParser(description='Learning a semantic parser for single domain')
    parser.add_argument('--single-domain', dest='single_domain', help='a domain used during training')
    args, unknown = parser.parse_known_args()

    return vars(args)


def _get_train_domains():
    arg_dict = parse_args()
    train_domains = [arg_dict['single_domain']]
    return train_domains


_config_specific = Environment(
    run_mode='train-default',
    train_domains=LazyEval(_get_train_domains)
)

config = Environment(distinct_pairs(chain(
    _config_general.items(),
    _config_specific.items()
)))
