
import re
import functools

from logic.formalism import Compiler
from logic.grammar import get_extra_ns

from kopl.kopl import KoPLEngine

from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.decorators import excepting

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
    def compile_tree(self, tree, tolerant=False):
        program = eval_lissp(tree.get_expr_str(), extra_ns=self.extra_ns)
        if tolerant:
            return excepting(Exception, runtime_exception_handler)(program)
        else:
            return program


NO_DENOTATION = object()

def runtime_exception_handler(exception):
    return NO_DENOTATION


def postprocess_denotation(denotation):
    if isinstance(denotation, list):
        new_denotation = [str(_) for _ in denotation]
    else:
        assert isinstance(denotation, (str, int, float))
        new_denotation = str(denotation)
    return new_denotation


def postprocess_answer(answer):
    '''
    Modify the result from `postprocess_denotation`
    '''

    if answer is NO_DENOTATION:
        new_answer = 'no'
    elif isinstance(answer, list) and len(answer) > 0:
        new_answer = answer[0]
    elif isinstance(answer, list) and len(answer) == 0:
        new_answer = 'None'
    else:
        new_answer = answer
    return new_answer


class KoPLContext(KoPLEngine):
    pass


class KoPLDebugContext(KoPLEngine):
    pass


def _initialize_debugging_context():
    def _debug_decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                breakpoint()
                result = func(*args, **kwargs)
                # print(func.__qualname__)
                # print(e.args)
            print('Error occured')

        return new_func


    _kopl_function_regex = re.compile(r"[A-Z]([a-zA-Z]*)")

    for attribute in dir(KoPLDebugContext):
        if _kopl_function_regex.match(attribute) is not None:
            obj = getattr(KoPLDebugContext, attribute)
            setattr(KoPLDebugContext, attribute, _debug_decorator(obj))


_initialize_debugging_context()
