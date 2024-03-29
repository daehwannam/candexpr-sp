
import itertools

from dhnamlib.pylib.klass import Interface

from splogic.seq2seq.transfer import TokenProcessing

from .kopl_interface import transfer as kopl_transfer


class KQAProTokenProcessing(TokenProcessing):
    interface = Interface(TokenProcessing)

    get_token_act_type = staticmethod(interface.implement(
        kopl_transfer.get_token_act_type))

    @staticmethod
    @interface.implement
    def get_non_distinctive_nl_token_act_type(token_value):
        return tuple(itertools.chain(
            kopl_transfer.default_nl_token_union_type,
            ['vp-quantity', 'vp-date', 'vp-year']))

    labeled_logical_form_to_action_seq = staticmethod(interface.implement(
        kopl_transfer.kopl_to_action_seq))
