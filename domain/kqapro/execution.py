
import re
import functools
import types

from logic.formalism import Compiler
from logic.grammar import get_extra_ns
from logic.function import make_call_limited

# from kopl.kopl import KoPLEngine

from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.decoration import excepting, running, construct

from dhnamlib.hissplib.macro import prelude
from dhnamlib.hissplib.compile import eval_lissp
from dhnamlib.hissplib.operation import import_operators

from .kopl_context import KoPLContext


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


def runtime_exception_handler(exception: Exception):
    return NO_DENOTATION


def invalid_program(*args, **kwargs):
    '''
    This function is used as the output program of parsing when the final paring states are incomplete.
    '''
    return NO_DENOTATION


def postprocess_denotation(denotation):
    if isinstance(denotation, list):
        new_denotation = [str(_) for _ in denotation]
    else:
        assert isinstance(denotation, (str, int, float))
        new_denotation = str(denotation)
    return new_denotation


def postprocess_prediction(answer):
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


# # MAX_NUM_LOOPS = 1000000
# MAX_NUM_LOOPS = 1000


class OverMaxNumIterationsError(Exception):
    pass


class CountingKoPLContext(KoPLContext):
    def __init__(self, context, max_num_iterations=1000000):
        self.kb = context.kb
        self._count = 0
        self._max_num_iterations = max_num_iterations

    def _iterate(self, iterator):
        for item in iterator:
            self._count += 1
            yield item
            if self._count >= self._max_num_iterations:
                raise OverMaxNumIterationsError


get_counting_context = CountingKoPLContext


_kopl_function_regex = re.compile(r"[A-Z]([a-zA-Z]*)")


@construct(types.SimpleNamespace, from_kwargs=True)
def get_call_limited_context(context: KoPLContext, max_num_calls=100):
    call_limited = make_call_limited(max_num_calls)
    for attribute in dir(context):
        if _kopl_function_regex.match(attribute) is not None:
            obj = getattr(context, attribute)
            if callable(obj):
                yield attribute, call_limited(obj)


class KoPLDebugContext(KoPLContext):
    pass


@running
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
            assert callable(obj)
            setattr(KoPLDebugContext, attribute, _debug_decorator(obj))
