
from dhnamlib.pylib.context import Scope
from dhnamlib.pylib.decorators import Register
from dhnamlib.pylib.filesys import json_load

# from transformers import BartTokenizer


# tokenizer = BartTokenizer.from_pretrained(
#     './pretrained/bart-base',
#     add_prefix_space=True)
# reduce_token = '<reduce>'
# tokenizer.add_tokens([reduce_token], special_tokens=True)

config = Scope(
    kb=json_load('./dataset/kb.json'),
    pretrained_model_name_or_path='./pretrained/bart-base',
    register=Register(strategy='conditional'),
    # tokenizer=tokenizer,
    # reduce_token=reduce_token,
)
