
import torch

from dhnamlib.pylib.torchlib.dnn import pad_sequence, SimpleDataset
from dhnamlib.pylib.iteration import dicts2pairs, keys2items
from dhnamlib.pylib.decoration import construct


def make_collate(pad_token_id):
    @construct(dict)
    def collate(examples):
        batched_example = dict(dicts2pairs(examples))
        for k, v in keys2items(batched_example, ['utterance_token_ids', 'action_ids']):
            yield k, torch.tensor(pad_sequence(v, pad_token_id), dtype=torch.int64)
        yield from keys2items(batched_example, ['answer', 'answer_by_program'])

    return collate


def make_data_loader(encoded_dataset, pad_token_id, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        SimpleDataset(encoded_dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate(pad_token_id),
    )
