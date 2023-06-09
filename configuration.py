
from dhnamlib.pylib.context import Scope, LazyEval
from dhnamlib.pylib.decorators import Register
from dhnamlib.pylib.filesys import json_load
from dhnamlib.pylib.iteration import apply_recursively

from logic.grammar import read_grammar

from domain.kopl.execution import KoPLContext, KoPLDebugContext

_kb_file_path = './dataset/kopl/kb.json'
_train_set_file_path = './dataset/kopl/train.json'
_val_set_file_path = './dataset/kopl/val.json'
_test_set_file_path = './dataset/kopl/test.json'

_train_action_seqs_file_path = './processed/kopl/train_action_seqs.jsonl'

_grammar_file_path = './domain/kopl/grammar.lissp'
_pretrained_model_name_or_path = './pretrained/bart-base'

DEBUG = True
if DEBUG:
    context_cls = KoPLContext
else:
    context_cls = KoPLDebugContext

def make_grammar():
    from domain.kopl.grammar import KoPLGrammar
    return read_grammar(_grammar_file_path, grammar_cls=KoPLGrammar)
    

config = Scope(
    register=Register(strategy='conditional'),

    kb=LazyEval(lambda: json_load(_kb_file_path)),
    train_set=LazyEval(lambda: json_load(_train_set_file_path)),
    val_set=LazyEval(lambda: json_load(_val_set_file_path)),
    test_set=LazyEval(lambda: json_load(_test_set_file_path)),

    grammar=LazyEval(make_grammar),
    context=LazyEval(lambda: context_cls(apply_recursively(config.kb))),
    
    pretrained_model_name_or_path=_pretrained_model_name_or_path,
    train_action_seqs_file_path=_train_action_seqs_file_path
)
