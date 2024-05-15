
from itertools import chain

from dhnamlib.pylib.klass import override, subclass
from dhnamlib.pylib.iteration import split_by_lengths

from splogic.base.execution import LazyExecutor

from overnight.domain_overnight import OvernightDomain, set_evaluator_path


set_evaluator_path('./overnight/evaluator')


@subclass
class OvernightExecutor(LazyExecutor):
    def __init__(self, domain):
        self.domain = domain
        self.evaluator = OvernightDomain(domain)

    @override
    def _process(self):
        lazy_results, programs_tuple, contexts_tuple = zip(*self._postponed_batch_groups)
        batch_sizes = tuple(map(len, programs_tuple))
        all_programs = tuple(chain(*programs_tuple))
        # all_contexts = tuple(chain(*contexts_tuple))

        all_denotations = self.evaluator(all_programs)
        denotations_tuple = split_by_lengths(all_denotations, batch_sizes)

        for lazy_result, denotations in zip(lazy_results, denotations_tuple):
            lazy_result._set_values(denotations)
