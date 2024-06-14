
from dhnamlib.pylib.context import Environment, LazyEval

from domain.kqapro.configuration import _make_ablation_grammar

import configuration

# cnlts: common_nl_token_seq


config = Environment(
    grammar=LazyEval(lambda: _make_ablation_grammar(using_common_nl_token_seq=True)),
    using_arg_candidate=False,
    constrained_decoding=False,

    encoded_train_set=LazyEval(lambda: configuration.config.encoded_cnlts_train_set),
    encoded_val_set=LazyEval(lambda: configuration.config.encoded_cnlts_val_set),

    shuffled_encoded_train_set=LazyEval(lambda: configuration.config.shuffled_encoded_cnlts_train_set),
    shuffled_encoded_val_set=LazyEval(lambda: configuration.config.shuffled_encoded_cnlts_val_set),
)
