import math

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

from ...common.extra.train_set_portion import compute_num_epoch_repeats


OVERNIGHT_TRIAN_SET_SIZE = 8751
# `OVERNIGHT_TRIAN_SET_SIZE` is the number of training examples of all domains.
# The examples used for validation are not counted.

def _make_config():
    return Environment(
        num_epoch_repeats=LazyEval(lambda: compute_num_epoch_repeats(
            ratio=len(configuration.config.encoded_weaksup_pretraining_set) / OVERNIGHT_TRIAN_SET_SIZE,
            epoch_repeat_strategy='linear'
        )),
        encoded_train_set=LazyEval(lambda: configuration.config.encoded_weaksup_pretraining_set),
    )


config = _make_config()
