
import re
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
    return kopl_function_regex.search(expr).group(1)
