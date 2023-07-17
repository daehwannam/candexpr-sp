
import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval


def _extract_dataset_portion(dataset, percent):
    # ratio = percent / 100
    num_examples = round(len(dataset) * percent / 100)
    return dataset[:num_examples]


def make_config(percent):
    assert 0 < percent <= 100
    num_epoch_repeats = 100 / percent

    return Environment(
        encoded_train_set=LazyEval(lambda: _extract_dataset_portion(
            configuration.config.shuffled_encoded_train_set,
            percent)),
        num_epoch_repeats=num_epoch_repeats
    )
