
import itertools
import argparse

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import not_none_valued_pairs
from dhnamlib.pylib.text import parse_bool

from .train_general import get_model_learning_dir_path
from .train_general import config as train_general_config


search_config = Environment(
    search_batch_size=8,
    num_search_beams=8,
    max_search_optim_loops=float('inf'),
)


optim_config = Environment(
    # Imported values
    train_general_config.items(),

    # Updated values
    num_train_epochs=16,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Learning a semantic parser form weak supervision')
    parser.add_argument('--pretrained-model-path', dest='pretrained_model_path', help='a path to a pretrained model')
    parser.add_argument('--resuming', dest='resuming', type=parse_bool, default=False, help='whether to resume learning')
    args, unknown = parser.parse_known_args()

    return vars(args)


config = Environment(not_none_valued_pairs(
    itertools.chain(
        search_config.items(),
        optim_config.items(),
        parse_args().items()
    ),

    # Others
    run_mode='search-train',
))
