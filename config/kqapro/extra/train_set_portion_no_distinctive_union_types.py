from itertools import chain
import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

from .train_set_portion import make_config_from_cmd

config = Environment(chain(
    make_config_from_cmd().items(),
    Environment(
        using_distinctive_union_types=False
    ).items(),
))
