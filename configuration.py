
import copy

from dhnamlib.pylib.context import Scope
from dhnamlib.pylib.decorators import Register
from dhnamlib.pylib.filesys import json_load
from dhnamlib.pylib.iteration import clone_recursively

from kopl.kopl import KoPLEngine

# from transformers import BartTokenizer


# tokenizer = BartTokenizer.from_pretrained(
#     './pretrained/bart-base',
#     add_prefix_space=True)
# reduce_token = '<reduce>'
# tokenizer.add_tokens([reduce_token], special_tokens=True)

_raw_kb = json_load('./dataset/kb.json')

# # test ------------------
# from dhnamlib.pylib.time import TimeMeasure
# import json
# with TimeMeasure() as tm:
#     _raw_kb = json_load('./dataset/kb.json')
# print(tm.interval)
# with TimeMeasure() as tm:
#     clone_recursively(_raw_kb)
# print(tm.interval)
# with TimeMeasure() as tm:
#     json.loads(json.dumps(_raw_kb))
# print(tm.interval)
# with TimeMeasure() as tm:
#     copy.deepcopy(_raw_kb)
# print(tm.interval)
# # ------------------ test

config = Scope(
    kb=_raw_kb,
    engine=KoPLEngine(clone_recursively(_raw_kb)),
    # engine=KoPLEngine(copy.deepcopy(_raw_kb)),
    pretrained_model_name_or_path='./pretrained/bart-base',
    register=Register(strategy='conditional'),
    # tokenizer=tokenizer,
    # reduce_token=reduce_token,
)
