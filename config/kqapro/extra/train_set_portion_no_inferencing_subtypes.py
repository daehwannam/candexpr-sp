from itertools import chain
import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

from .train_set_portion import make_config_from_cmd

config = Environment(chain(
    make_config_from_cmd('shuffled_encoded_strict_train_set').items(),
    Environment(
        inferencing_subtypes=False,
        encoded_val_set=LazyEval(lambda: configuration.config.encoded_strict_val_set)).items(),
))
