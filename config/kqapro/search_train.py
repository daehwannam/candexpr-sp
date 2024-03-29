
import itertools
import argparse

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import not_none_valued_pairs
from dhnamlib.pylib.text import parse_bool

from .train_general import get_model_learning_dir_path
from .train_general import config as train_general_config


common_config = Environment(
    # max_search_optim_loops=float('inf'),
    max_search_optim_loops=16,
)


search_config = Environment(
    search_batch_size=8,
    num_search_beams=8,
)

_TRAIN_BATCH_SIZE = 16

optim_config = Environment(
    # Imported values
    train_general_config.items(),

    # Updated values
    num_train_epochs=8,
    train_batch_size=_TRAIN_BATCH_SIZE,
    train_max_num_action_seqs=_TRAIN_BATCH_SIZE,
    val_batch_size=64,
    using_scheduler=False,
    learning_rate=2e-5,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Learning a semantic parser form weak supervision')
    parser.add_argument('--pretrained-model-path', dest='pretrained_model_path', help='a path to a pretrained model')
    parser.add_argument('--resuming', dest='resuming', type=parse_bool, default=False, help='whether to resume learning')
    args, unknown = parser.parse_known_args()

    return vars(args)


config = Environment(not_none_valued_pairs(
    itertools.chain(
        common_config.items(),
        search_config.items(),
        optim_config.items(),
        parse_args().items()
    ),

    # Others
    run_mode='search-train',
))
