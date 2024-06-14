
from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

from domain.overnight.configuration import _load_val_set


def _get_repeated_val_set(num_repeats=5):
    val_set = _load_val_set()
    repeated_val_set = val_set * num_repeats
    return repeated_val_set


config = Environment(
    encoded_val_set=LazyEval(_get_repeated_val_set),
)
