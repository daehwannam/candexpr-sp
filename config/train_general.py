
import os

from dhnamlib.pylib.context import Environment
# from dhnamlib.pylib.context import LazyEval

from util.time import current_date_str


_model_dir_root_path = './model-instance'


config = Environment(
    mode='train',
    model_dir_path=os.path.join(_model_dir_root_path, current_date_str),

    learning_rate=3e-5,
    weight_decay=1e-5,
    batch_size=16,
    num_train_epochs=25,
    # num_warmup_steps=...,
)

raise NotImplementedError('Add more config attributes here ')


# model_instance_dir_path = './model-instance'


# config = Environment(
#     mode='train',
#     model_dir_path='./model-instance',
#     finetuned_model_path=os.path.join(model_instance_dir_path, )

#     learning_rate=3e-5,
#     weight_decay=1e-5,
#     batch_size=16,
#     num_train_epochs=25,
#     # num_warmup_steps=...,
# )
