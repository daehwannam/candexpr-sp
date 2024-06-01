
from dhnamlib.pylib.context import Environment, LazyEval

import configuration


config = Environment(
    test_domains=LazyEval(lambda: configuration.config.all_domains)
)
