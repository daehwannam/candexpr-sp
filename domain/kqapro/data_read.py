
import torch
import itertools

from dhnamlib.pylib.torchlib.dnn import (
    pad_sequence, id_tensor_to_mask, batch_sequence_tensors,
    except_last_tokens, except_first_tokens
)
from dhnamlib.pylib.torchlib.data_processing import SimpleDataset, EpochRepeatingDataLoader
# from dhnamlib.pylib.iteration import keys2items
from dhnamlib.pylib.iteration import (
    dicts2pairs, not_none_valued_pairs, merge_dicts, unique, not_none_valued_dict, slice_by_max_size, lengths_to_slices
)
from dhnamlib.pylib.decoration import deprecated, construct
from dhnamlib.pylib.structure import LazyDict, LazyEval
from dhnamlib.pylib.function import identity
from dhnamlib.pylib.statistics import shuffled


def make_collate(decoder_start_token_id, pad_token_id, max_num_action_seqs):
    def make_batched_long_tensor(lists):
        return torch.tensor(pad_sequence(lists, pad_token_id), dtype=torch.int64)

    def batched_mask_init_fn(batched_mask):
        batched_mask[:, :, pad_token_id] = 1

    def make_batched_mask_tensor(masks):
        return batch_sequence_tensors(
            tuple(map(torch.tensor, masks)),
            padding_value=0,
            init_fn=batched_mask_init_fn)

    def prepend_decoder_start_token_id(batched_tensor):
        num_seqs = batched_tensor.shape[0]
        return torch.concat([torch.ones(num_seqs, 1, dtype=torch.int64) * decoder_start_token_id,
                             batched_tensor],
                            dim=1)

    @construct(tuple)
    def make_ws_sub_batches(batched_example):
        sub_batch_slices = slice_by_max_size(
            batched_example['action_id_seq_group'],
            size_fn=len,
            max_size=max_num_action_seqs
        )
        sub_batched_examples = tuple(
            dict([k, v[sub_batch_slice]] for k, v in batched_example.items())
            for sub_batch_slice in sub_batch_slices)

        for sub_batched_example in sub_batched_examples:
            action_id_seq_group_len = tuple(map(len, sub_batched_example['action_id_seq_group']))
            utterance_token_ids = LazyEval(lambda: make_batched_long_tensor(tuple(itertools.chain(
                *([token_id_seq] * group_len
                  for token_id_seq, group_len in zip(sub_batched_example['utterance_token_ids'],
                                                     action_id_seq_group_len))))))
            attention_mask = LazyEval(lambda: id_tensor_to_mask(utterance_token_ids.get(), pad_token_id))

            _all_action_id_seq_group = make_batched_long_tensor(tuple(itertools.chain(*sub_batched_example['action_id_seq_group'])))
            all_action_id_seq_group = prepend_decoder_start_token_id(_all_action_id_seq_group)
            decoder_input_ids = LazyEval(lambda: except_last_tokens(all_action_id_seq_group))
            labels = LazyEval(lambda: except_first_tokens(all_action_id_seq_group))

            yield LazyDict(
                action_id_seq_group_len=action_id_seq_group_len,
                utterance_token_ids=utterance_token_ids,
                attention_mask=attention_mask,
                all_action_id_seq_group=all_action_id_seq_group,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )

    def collate(examples):
        batch_size = len(examples)
        batched_example = dict(dicts2pairs(*examples))

        example_id = batched_example['example_id']
        utterance_token_ids = LazyEval(lambda: make_batched_long_tensor(batched_example['utterance_token_ids']))
        attention_mask = LazyEval(lambda: id_tensor_to_mask(utterance_token_ids.get(), pad_token_id))

        if 'action_ids' in batched_example:
            _action_ids = make_batched_long_tensor(batched_example['action_ids'])
            action_ids = prepend_decoder_start_token_id(_action_ids)
            # `action_ids[some_index]` is a sequence like
            # [decoder_start_token_id, bos_token_id, ..., some-token-id, ..., eos_token_id, pad_token_id, pad_token_id, ...]

            # except the last tokens (either PAD or EOS)
            decoder_input_ids = LazyEval(lambda: except_last_tokens(action_ids))

            # except the first tokens (decoder_start_token_id)
            labels = LazyEval(lambda: except_first_tokens(action_ids))
        else:
            action_ids = None
            decoder_input_ids = None
            labels = None

        answer = batched_example.get('answer')
        answer_by_program = batched_example.get('answer_by_program')

        if 'action_id_seq_group' in batched_example:
            ws_sub_batches = LazyEval(lambda: make_ws_sub_batches(batched_example))
        else:
            ws_sub_batches = None

        return LazyDict(not_none_valued_pairs(
            example_id=example_id,
            utterance_token_ids=utterance_token_ids,
            action_ids=action_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            answer=answer,
            answer_by_program=answer_by_program,
            # **mask_data_dict
            ws_sub_batches=ws_sub_batches,
        ))

    return collate


def make_data_loader(
        encoded_dataset, encoded_mask_dataset=None, *, decoder_start_token_id, pad_token_id,
        batch_size=None, shuffle, num_epoch_repeats=1, max_num_action_seqs=None
):
    if encoded_mask_dataset is None:
        _encoded_dataset = encoded_dataset
    else:
        assert len(encoded_dataset) == len(encoded_mask_dataset)
        _encoded_dataset = tuple(merge_dicts(examples, merge_fn=unique)
                                 for examples in zip(encoded_dataset, encoded_mask_dataset))

    data_loader = torch.utils.data.DataLoader(
        SimpleDataset(_encoded_dataset),
        **not_none_valued_dict(
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=make_collate(decoder_start_token_id, pad_token_id, max_num_action_seqs),
        ))

    if num_epoch_repeats == 1:
        return data_loader
    else:
        return EpochRepeatingDataLoader(data_loader, num_epoch_repeats=num_epoch_repeats)


# @deprecated
# def split_into_weaksup_sub_batches(batch_iterator, max_num_action_seqs):
#     from dhnamlib.pylib.iteration import generate_variable_sized_slices

#     keys = ['ws_utterance_token_ids', 'ws_attention_mask', 'ws_decoder_input_ids', 'ws_labels', 'action_id_seq_group_len']

#     for batch in batch_iterator:
#         group_slices = generate_variable_sized_slices(
#             items=batch['action_id_seq_group_len'],
#             size_fn=identity,
#             max_size=max_num_action_seqs
#         )
#         last_idx = len(group_slices) - 1

#         for idx, group_slice in enumerate(group_slices):
#             last_in_batch = (idx == last_idx)
#             sub_batch = dict([key, batch[key][group_slice]] for key in keys)

#             yield last_in_batch, sub_batch


# @deprecated
# def no_split_into_sub_batches(batch_iterator, max_num_action_seqs):
#     last_in_batch = True
#     for batch in batch_iterator:
#         yield last_in_batch, batch


# @deprecated
# def to_sub_batches(batch, max_num_action_seqs):
#     keys = ['ws_utterance_token_ids', 'ws_attention_mask', 'ws_decoder_input_ids', 'ws_labels']
#     group_len_key = 'action_id_seq_group_len'

#     group_length_tuples = tuple(split_by_max_size(
#         items=batch[group_len_key],
#         size_fn=identity,
#         max_size=max_num_action_seqs
#     ))
#     group_length_sums = tuple(map(sum, group_length_tuples))
#     for sub_batch_slice, group_length_tuple in zip(lengths_to_slices(group_length_sums), group_length_tuples):
#         sub_batch = dict([key, batch[key][sub_batch_slice]] for key in keys)
#         sub_batch.update([[group_len_key, group_length_tuple]])
#         yield sub_batch
