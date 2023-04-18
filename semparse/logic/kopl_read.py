
from itertools import chain
from pprint import pprint as pp

from logic.dsl import read_dsl


def kopl_to_recursive_form(labeled_kopl_program):
    def parse(func_call_idx):
        func_call = labeled_kopl_program[func_call_idx]
        return dict(function=func_call['function'],
                    inputs=func_call['inputs'],
                    dependencies=list(map(parse, func_call['dependencies'])))

    return parse(len(labeled_kopl_program) - 1)


kopl_function_regex = re.compile("engine\.([a-zA-Z]+)")


def extract_kopl_func(expr):
    # e.g. expr == "(engine.VerifyDate @2 @0 @1)"
    return kopl_function_regex.search(expr).group(1)


def recursive_form_to_action_seq(dsl, recursive_form):
    action_seq = []
    def recurse(form):
        pass
        for sub_form in form['dependencies']:
            pass
    pass


#
# - A function
#   - No inputs, no dependencies -> terminal action
# - An input
#   - op-*
#     #+begin_src python
#     (('op-eq', '='), ('op-ne', '!='), ('op-lt', '<'), ('op-gt', '>'))
#     #+end_src
#   - direction-*
#     #+begin_src python
#     (('direction-forward', 'forward'), ('direction-backward', 'backward'))
#     #+end_src
#   - op-* 2
#     #+begin_src python
#     (('op-st', 'smaller'), ('op-gt', 'greater'))
#     #+end_src
#   - op-* 3
#     #+begin_src python
#     (('op-min', 'min'), ('op-max', 'max'))
#     #+end_src
#   - Others -> constants and tokens
#


# KB -> keyword-type pairs


if __name__ == '__main__':
    dsl = read_dsl('./dsl/kopl/dsl')
    pass
