
from itertools import chain

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import distinct_pairs
from dhnamlib.pylib.structure import AttrDict

from .train_general import config as _config_general
import configuration


def _make_grammar_with_no_dut():
    with configuration.config.let(using_distinctive_union_types=False):
        return configuration._make_grammar()


_config_specific = Environment(
    # specific config here
    run_mode='train-for-multiple-decoding-strategies',
    decoding_strategy_configs=[
        AttrDict(
            decoding_strategy_name='full-constraints',
            constrained_decoding=True,
            using_arg_candidate=True,
        ),
        AttrDict(
            decoding_strategy_name='no-arg-candidate',
            constrained_decoding=True,
            using_arg_candidate=False,
        ),
        AttrDict(
            decoding_strategy_name='no-ac-no-dut',
            constrained_decoding=True,
            using_arg_candidate=False,
            grammar_lazy_obj=LazyEval(_make_grammar_with_no_dut),
        ),
        AttrDict(
            decoding_strategy_name='no-constrained-decoding',
            constrained_decoding=False,
            using_arg_candidate=False,
            grammar_lazy_obj=LazyEval(_make_grammar_with_no_dut),
        ),
    ]
)

config = Environment(chain(
    distinct_pairs(chain(
        _config_general.items(),
        _config_specific.items()))
))
