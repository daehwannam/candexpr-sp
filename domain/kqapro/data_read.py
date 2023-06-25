
import torch

from dhnamlib.pylib.torchlib.dnn import pad_sequence, SimpleDataset
# from dhnamlib.pylib.iteration import keys2items
from dhnamlib.pylib.iteration import dicts2pairs, filter_dict_values, is_not_none
# from dhnamlib.pylib.decoration import construct
from dhnamlib.pylib.structure import LazyDict, LazyEval
from dhnamlib.pylib.torchlib.dnn import id_tensor_to_mask
# from dhnamlib.pylib.function import identity


def make_collate(pad_token_id):
    def make_batched_tensor(lists):
        return torch.tensor(pad_sequence(lists, pad_token_id), dtype=torch.float32)

    def collate(examples):
        batched_example = dict(dicts2pairs(examples))

        utterance_token_ids = make_batched_tensor(batched_example['utterance_token_ids'])
        action_ids = make_batched_tensor(batched_example['action_ids'])

        attention_mask = id_tensor_to_mask(utterance_token_ids, pad_token_id)

        # except the last tokens (either PAD or EOS)
        decoder_input_ids = LazyEval(lambda: action_ids[:, :-1].contiguous())

        # except the first tokens (BOS)
        labels = LazyEval(lambda: action_ids[:, 1:].contiguous())

        answer_data_dict = dict(filter_dict_values(
            is_not_none,
            dict(
                answer=batched_example.get('answer'),
                answer_by_program=batched_example.get('answer_by_program'))))

        return LazyDict(
            utterance_token_ids=utterance_token_ids,
            action_ids=action_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **answer_data_dict
        )

    return collate


def make_data_loader(encoded_dataset, pad_token_id, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        SimpleDataset(encoded_dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate(pad_token_id),
    )
