
from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.debug import NIE

from splogic.base.grammar import read_grammar


_pretrained_model_name_or_path = 'facebook/bart-base'
# _OVERNIGHT_DOMAINS = ('basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork')
_OVERNIGHT_DOMAINS = ('calendar', 'blocks', 'housing', 'restaurants', 'publications', 'recipes', 'socialnetwork', 'basketball')

_raw_dataset_dir_path = './dataset/overnight'
_augmented_dataset_dir_path = './processed/overnight/augmented'
_encoded_dataset_dir_path = './processed/overnight/encoded'

_grammar_file_path = './domain/overnight/grammar.lissp'

_NO_CONTEXT = object()


def _make_grammar():
    from .grammar import OvernightGrammar
    return read_grammar(
        _grammar_file_path,
        grammar_cls=OvernightGrammar,
        grammar_kwargs=dict(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path
        ))


config = Environment(
    using_arg_candidate=True,
    using_arg_filter=False,
    inferencing_subtypes=True,
    using_distinctive_union_types=True,
    pretrained_model_name_or_path=_pretrained_model_name_or_path,
    context=_NO_CONTEXT,
    grammar=LazyEval(_make_grammar),

    all_domains = _OVERNIGHT_DOMAINS,
    train_domains = LazyEval(NIE),
    test_domains = LazyEval(NIE),

    raw_dataset_dir_path=_raw_dataset_dir_path,
    augmented_dataset_dir_path=_augmented_dataset_dir_path,
)
