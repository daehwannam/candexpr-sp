
import os
import argparse

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
        os.path.join(_model_learning_dir_root_path, initial_date_str)


config = Environment(not_none_valued_pairs(dict(
    model_learning_dir_path=get_model_learning_dir_path(),

    learning_rate=3e-5,
    # learning_rate=1e-3,
    adam_epsilon=1e-8,
    weight_decay=1e-5,
    train_batch_size=16,
    val_batch_size=LazyEval(lambda: config.train_batch_size * 4),
    # val_batch_size=1,
    num_train_epochs=25,
    using_scheduler=True,
    num_warmup_epochs=LazyEval(lambda: config.num_train_epochs / 10),
    max_grad_norm=1,
    saving_optimizer=False,
).items()))

# model_instance_dir_path = './model-instance'


# config = Environment(
#     mode='train',
#     model_learning_dir_path='./model-instance',
#     finetuned_model_path=os.path.join(model_instance_dir_path, )

#     learning_rate=3e-5,
#     weight_decay=1e-5,
#     batch_size=16,
#     num_train_epochs=25,
#     # num_warmup_steps=...,
# )
