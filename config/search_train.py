
import os

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import not_none_valued_pairs

from configuration import has_model_learning_dir_path
from utility.time import initial_date_str


_model_learning_dir_root_path = './model-instance'


def get_model_learning_dir_path():
    if has_model_learning_dir_path():
        return None
    else:
        return os.path.join(_model_learning_dir_root_path, initial_date_str)


config = Environment(not_none_valued_pairs(
    model_learning_dir_path=get_model_learning_dir_path(),

    #
    # Search
    #
    search_batch_size=8,
    num_search_beams=8,

    #
    # Optimization
    #
    # learning_rate=3e-5,
    learning_rate=2e-5,
    adam_epsilon=1e-8,
    weight_decay=1e-5,
    train_batch_size=16,
    val_batch_size=LazyEval(lambda: config.train_batch_size * 4),
    # num_train_epochs=25,
    num_train_epochs=16,
    # using_scheduler=True,
    using_scheduler=False,
    num_warmup_epochs=LazyEval(lambda: config.num_train_epochs / 10),
    max_grad_norm=1,
    # saving_optimizer=False,
    patience=1,
))
