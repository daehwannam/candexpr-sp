
from logic.formalism import Compiler
from logic.grammar import get_extra_ns

from dhnamlib.pylib.klass import Interface

from dhnamlib.hissplib.macro import prelude
from dhnamlib.hissplib.compile import eval_lissp
from dhnamlib.hissplib.operation import import_operators


prelude()  # used for eval_lissp
import_operators()  # used for eval_lissp


class KoPLCompiler(Compiler):
    interface = Interface(Compiler)

    def __init__(self):
        bindings = [['postprocess-denotation', postprocess_denotation]]
        self.extra_ns = get_extra_ns(bindings)

    @interface.implement
    def compile_trees(self, trees):
        return tuple(eval_lissp(tree.get_expr_str, extra_ns=self.extra_ns)
                     for tree in trees)


def postprocess_denotation(denotation):
    if isinstance(denotation, list):
        new_denotation = [str(_) for _ in denotation]
    else:
        new_denotation = str(denotation)
    return new_denotation


def postprocess_answer(answer):
    if answer is None:
        new_answer = 'no'
    elif isinstance(answer, list) and len(answer) > 0:
        new_answer = answer[0]
    elif isinstance(answer, list) and len(answer) == 0:
        new_answer = 'None'
    else:
        new_answer = answer
    return new_answer
