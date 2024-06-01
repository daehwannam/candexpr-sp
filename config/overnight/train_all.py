
from itertools import chain

from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.iteration import distinct_pairs

import configuration

from .train_general import config as _config_general


_config_specific = Environment(
    train_domains=LazyEval(lambda: configuration.config.all_domains),
)

config = Environment(distinct_pairs(chain(
    _config_general.items(),
    _config_specific.items()
)))
