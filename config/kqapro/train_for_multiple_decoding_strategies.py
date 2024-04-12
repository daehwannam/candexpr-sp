
from itertools import chain

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import distinct_pairs
from dhnamlib.pylib.structure import AttrDict
# from dhnamlib.pylib.decoration import deprecated

from domain.kqapro import configuration as kqapro_configuration

from .train_general import config as _config_general


# @deprecated
def _make_grammar_with_no_dut():
    with kqapro_configuration.config.let(using_distinctive_union_types=False):
        return kqapro_configuration._make_grammar()


_config_specific = Environment(
    # specific config here
    run_mode='train-for-multiple-decoding-strategies',
    decoding_strategy_configs=[
        AttrDict(
            decoding_strategy_name='full-constraints',
            constrained_decoding=True,
            using_arg_candidate=True,
            using_distinctive_union_types=True,
        ),
        AttrDict(
            decoding_strategy_name='no-arg-candidate',
            constrained_decoding=True,
            using_arg_candidate=False,
            using_distinctive_union_types=True,
        ),
        AttrDict(
            decoding_strategy_name='no-ac-no-dut',
            constrained_decoding=True,
            using_arg_candidate=False,
            using_distinctive_union_types=False,
            # grammar_lazy_obj=LazyEval(kqapro_configuration._make_grammar),
            grammar_lazy_obj=LazyEval(_make_grammar_with_no_dut),
        ),
        AttrDict(
            decoding_strategy_name='no-constrained-decoding',
            constrained_decoding=False,
            using_arg_candidate=False,
            using_distinctive_union_types=False,
            # grammar_lazy_obj=LazyEval(kqapro_configuration._make_grammar),
            grammar_lazy_obj=LazyEval(_make_grammar_with_no_dut),
        ),
    ]
)

config = Environment(chain(
    distinct_pairs(chain(
        _config_general.items(),
        _config_specific.items()))
))
