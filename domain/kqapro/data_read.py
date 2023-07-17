
import torch

from dhnamlib.pylib.torchlib.dnn import pad_sequence, SimpleDataset, id_tensor_to_mask, batch_sequence_tensors
# from dhnamlib.pylib.iteration import keys2items
from dhnamlib.pylib.iteration import dicts2pairs, not_none_valued_pairs, merge_dicts, unique
# from dhnamlib.pylib.decoration import construct
from dhnamlib.pylib.structure import LazyDict, LazyEval
# from dhnamlib.pylib.function import identity


def make_collate(decoder_start_token_id, pad_token_id):
    def make_batched_long_tensor(lists):
        return torch.tensor(pad_sequence(lists, pad_token_id), dtype=torch.int64)

    def batched_mask_init_fn(batched_mask):
        batched_mask[:, :, pad_token_id] = 1

    def make_batched_mask_tensor(masks):
        return batch_sequence_tensors(
            tuple(map(torch.tensor, masks)),
            padding_value=0,
            init_fn=batched_mask_init_fn)

    def collate(examples):
        batch_size = len(examples)
        batched_example = dict(dicts2pairs(*examples))

        example_id = batched_example['example_id']
        utterance_token_ids = make_batched_long_tensor(batched_example['utterance_token_ids'])
        attention_mask = id_tensor_to_mask(utterance_token_ids, pad_token_id)

        if 'action_ids' in batched_example:
            _action_ids = make_batched_long_tensor(batched_example['action_ids'])
            action_ids = torch.concat([torch.ones(batch_size, 1, dtype=torch.int64) * decoder_start_token_id, _action_ids], dim=1)
            # `action_ids[some_index]` is a sequence like
            # [decoder_start_token_id, bos_token_id, ..., some-token-id, ..., eos_token_id, pad_token_id, pad_token_id, ...]

            # except the last tokens (either PAD or EOS)
            decoder_input_ids = LazyEval(lambda: action_ids[:, :-1].contiguous())

            # except the first tokens (decoder_start_token_id)
            labels = LazyEval(lambda: action_ids[:, 1:].contiguous())
        else:
            action_ids = None
            decoder_input_ids = None
            labels = None

        answer = batched_example.get('answer')
        answer_by_program = batched_example.get('answer_by_program')

        # mask_data_dict = dict(not_none_valued_pairs(
        #     softmax_mask=make_batched_mask_tensor(batched_example['softmax_mask']),
        #     nll_mask=make_batched_mask_tensor(batched_example['nll_mask'])))

        return LazyDict(not_none_valued_pairs(
            example_id=example_id,
            utterance_token_ids=utterance_token_ids,
            action_ids=action_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            answer=answer,
            answer_by_program=answer_by_program
            # **mask_data_dict
        ))

    return collate


def make_data_loader(encoded_dataset, encoded_mask_dataset=None, *, decoder_start_token_id, pad_token_id, batch_size, shuffle):
    if encoded_mask_dataset is None:
        _encoded_dataset = encoded_dataset
    else:
        assert len(encoded_dataset) == len(encoded_mask_dataset)
        _encoded_dataset = tuple(merge_dicts(examples, merge_fn=unique)
                                 for examples in zip(encoded_dataset, encoded_mask_dataset))

    return torch.utils.data.DataLoader(
        SimpleDataset(_encoded_dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate(decoder_start_token_id, pad_token_id),
    )
