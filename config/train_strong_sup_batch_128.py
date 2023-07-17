
from itertools import chain

from dhnamlib.pylib.context import Environment
# from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import distinct_pairs

from .train_general import config as _config_general


_config_specific = Environment(
    # specific config here
    run_mode='train-strong-sup',
)

_config_overriding = Environment(
    # overriding config here
    train_batch_size=128,
    num_train_epochs=200,
)

config = Environment(chain(
    distinct_pairs(chain(
        _config_general.items(),
        _config_specific.items())),
    _config_overriding.items()))
