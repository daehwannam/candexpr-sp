
import math

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval


KQAPRO_TRAN_SET_SIZE = 94376


def _compute_num_epoch_repeats(sub_dataset):
    assert len(sub_dataset) <= KQAPRO_TRAN_SET_SIZE
    return math.sqrt(KQAPRO_TRAN_SET_SIZE / len(sub_dataset))


def _make_config():
    sub_dataset_lazy_obj = LazyEval(lambda: configuration.config.encoded_weaksup_pretraining_set)
    return Environment(
        num_epoch_repeats=LazyEval(lambda: _compute_num_epoch_repeats(sub_dataset_lazy_obj.get())),
        num_used_train_examples=LazyEval(lambda: len(sub_dataset_lazy_obj.get())),
        encoded_train_set=LazyEval(sub_dataset_lazy_obj.get),
    )


config = _make_config()
