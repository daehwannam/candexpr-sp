
import torch
import itertools

from dhnamlib.pylib.torchlib.dnn import (
    pad_sequence, id_tensor_to_mask, batch_sequence_tensors,
    except_last_tokens, except_first_tokens
)
from dhnamlib.pylib.torchlib.data_processing import SimpleDataset, EpochRepeatingDataLoader, generate_variable_sized_slices
# from dhnamlib.pylib.iteration import keys2items
from dhnamlib.pylib.iteration import dicts2pairs, not_none_valued_pairs, merge_dicts, unique, not_none_valued_dict
from dhnamlib.pylib.decoration import deprecated
from dhnamlib.pylib.structure import LazyDict, LazyEval
from dhnamlib.pylib.function import identity
from dhnamlib.pylib.statistics import shuffled


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
        utterance_token_ids = LazyEval(lambda: make_batched_long_tensor(batched_example['utterance_token_ids']))
        attention_mask = LazyEval(lambda: id_tensor_to_mask(utterance_token_ids.get(), pad_token_id))

        def prepend_decoder_start_token_id(batched_tensor):
            num_seqs = batched_tensor.shape[0]
            return torch.concat([torch.ones(num_seqs, 1, dtype=torch.int64) * decoder_start_token_id,
                                 batched_tensor],
                                dim=1)

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

        if 'action_id_seq_group' in batched_example:
            action_id_seq_group_len = tuple(map(len, batched_example['action_id_seq_group']))
            ws_utterance_token_ids = LazyEval(lambda: make_batched_long_tensor(tuple(itertools.chain(
                *([token_id_seq] * group_len
                  for token_id_seq, group_len in zip(batched_example['utterance_token_ids'],
                                                     action_id_seq_group_len))))))
            ws_attention_mask = LazyEval(lambda: id_tensor_to_mask(ws_utterance_token_ids.get(), pad_token_id))

            _all_action_id_seq_group = make_batched_long_tensor(tuple(itertools.chain(*batched_example['action_id_seq_group'])))
            all_action_id_seq_group = prepend_decoder_start_token_id(_all_action_id_seq_group)
            ws_decoder_input_ids = LazyEval(lambda: except_last_tokens(all_action_id_seq_group))
            ws_labels = LazyEval(lambda: except_first_tokens(all_action_id_seq_group))
        else:
            action_id_seq_group_len = None
            ws_utterance_token_ids = None
            ws_attention_mask = None
            all_action_id_seq_group = None
            ws_decoder_input_ids = None
            ws_labels = None

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
            answer_by_program=answer_by_program,
            # **mask_data_dict
            action_id_seq_group_len=action_id_seq_group_len,
            ws_utterance_token_ids=ws_utterance_token_ids,
            ws_attention_mask=ws_attention_mask,
            all_action_id_seq_group=all_action_id_seq_group,
            ws_decoder_input_ids=ws_decoder_input_ids,
            ws_labels=ws_labels
        ))

    return collate


def make_data_loader(
        encoded_dataset, encoded_mask_dataset=None, *, decoder_start_token_id, pad_token_id,
        batch_size=None, shuffle, num_epoch_repeats=1,
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
            collate_fn=make_collate(decoder_start_token_id, pad_token_id),
        ))

    if num_epoch_repeats == 1:
        return data_loader
    else:
        return EpochRepeatingDataLoader(data_loader, num_epoch_repeats=num_epoch_repeats)


@deprecated
def split_into_weaksup_sub_batches(batch_iterator, max_num_batch_seqs):
    keys = ['ws_utterance_token_ids', 'ws_attention_mask', 'ws_decoder_input_ids', 'ws_labels', 'action_id_seq_group_len']

    for batch in batch_iterator:
        group_slices = generate_variable_sized_slices(
            data_source=batch['action_id_seq_group_len'],
            size_fn=identity,
            max_size=max_num_batch_seqs
        )
        last_idx = len(group_slices) - 1

        for idx, group_slice in enumerate(group_slices):
            last_in_batch = (idx == last_idx)
            sub_batch = dict([key, batch[key][group_slice]] for key in keys)

            yield last_in_batch, sub_batch


@deprecated
def no_split_into_sub_batches(batch_iterator, max_num_batch_seqs):
    last_in_batch = True
    for batch in batch_iterator:
        yield last_in_batch, batch


def to_sub_batches(batch, max_num_batch_seqs):
    keys = ['ws_utterance_token_ids', 'ws_attention_mask', 'ws_decoder_input_ids', 'ws_labels', 'action_id_seq_group_len']

    group_slices = generate_variable_sized_slices(
        data_source=batch['action_id_seq_group_len'],
        size_fn=identity,
        max_size=max_num_batch_seqs
    )

    for idx, group_slice in enumerate(group_slices):
        sub_batch = dict([key, batch[key][group_slice]] for key in keys)
        yield sub_batch
