
from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import not_none_valued_pairs

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


config = Environment(not_none_valued_pairs(
    search_config.items(),
    optim_config.items(),

    # Others
    run_mode='search-train',
))
