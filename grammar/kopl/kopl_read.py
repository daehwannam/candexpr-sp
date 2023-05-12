
import re
from kopl.kopl import KoPLEngine

# from itertools import chain
# from pprint import pprint as pp


def kopl_to_recursive_form(labeled_kopl_program):
    def parse(func_call_idx):
        func_call = labeled_kopl_program[func_call_idx]
        return dict(function=func_call['function'],
                    inputs=func_call['inputs'],
                    dependencies=list(map(parse, func_call['dependencies'])))

    return parse(len(labeled_kopl_program) - 1)


kopl_function_regex = re.compile(r"engine\.([a-zA-Z]+)")


def extract_kopl_func(expr):
    # e.g. expr == "(engine.VerifyDate @2 @0 @1)"
    match_obj = kopl_function_regex.search(expr)
    if match_obj:
        return match_obj.group(1)
    else:
        assert match_obj is None
        return None


def is_camel_case(s):
    # https://stackoverflow.com/a/10182901
    return s != s.lower() and s != s.upper() and "_" not in s


def generate_kopl_function_names():
    for attr in dir(KoPLEngine):
        if (
                is_camel_case(attr) and
                attr[0].upper() == attr[0] and
                callable(getattr(KoPLEngine, attr))
            ):
            yield attr
