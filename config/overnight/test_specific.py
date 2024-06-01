
import argparse
from functools import cache

from dhnamlib.pylib.context import Environment, LazyEval

import configuration


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


config = Environment(
    train_domains=LazyEval(_get_train_domains)
)
