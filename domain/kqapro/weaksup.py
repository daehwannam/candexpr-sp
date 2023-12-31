
from fractions import Fraction
import torch

from domain.kqapro.data_read import make_collate

from dhnamlib.pylib.multiprocessing import Processor, ArgGroup
from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.iteration import partition
from dhnamlib.pylib.mllib.learning import get_performance
from dhnamlib.pylib import filesys
from dhnamlib.pylib.mllib.learning import get_performance, get_init_status, update_status

from configuration import config

from . import learning
from .validation import get_consistent_action_id_seq_groups
from .data_read import make_data_loader
from .learning import optim_measures


def search_to_collect(
        grammar, compiler, model_path, devices, context, encoded_dataset, batch_size,
        num_beams, generation_max_length, constrained_decoding, using_arg_candidate,
):
    common_arg_group = ArgGroup(
        grammar=grammar,
        compiler=compiler,
        model_path=model_path,
        context=context,
        encoded_dataset=encoded_dataset,
        batch_size=batch_size,
        num_beams=num_beams,
        generation_max_length=generation_max_length,
        constrained_decodin=constrained_decoding,
        using_arg_candidate=using_arg_candidate,
        strict_postprocessing=False
    )

    realtime_num_correct = 0
    realtime_num_examples = 0

    def get_realtime_oracle_accuracy_percent():
        return realtime_num_correct * 100 / realtime_num_examples

    dataset_parts = tuple(partition(encoded_dataset, batch_size))
    weaksup_examples = []

    for dataset_part, action_id_seq_groups in config.utqdm(
            zip(dataset_parts,
                SearchProcessor.map(
                    coll=dataset_parts,
                    arg_groups=[common_arg_group.augment(device=device)
                                for device in devices])),
            unit='oracle_accuracy',
            update_fn=get_realtime_oracle_accuracy_percent,
            repr_format='{:5.2f}',
            init_repr='none',
            total=len(dataset_parts)
    ):
        for example, action_id_seq_group in zip(dataset_parts, action_id_seq_groups):
            realtime_num_examples += 1

            if len(action_id_seq_group) > 0:
                realtime_num_correct += 1

                weaksup_example = dict(
                    **example,
                    action_id_seq_group=action_id_seq_group
                )
                weaksup_examples.append(weaksup_example)

    assert realtime_num_examples == len(encoded_dataset)

    search_result = dict(
        weaksup_examples=weaksup_examples,
        performance=get_performance(
            oracle_accuracy=realtime_num_correct / realtime_num_examples,
            oracle_accuracy_fraction=Fraction(realtime_num_correct, realtime_num_examples)))

    return search_result


class SearchProcessor(Processor):
    interface = Interface(Processor)

    @interface.implement
    def initialize(
            self, grammar, compiler, model_path, device, context,
            num_beams, generation_max_length, constrained_decoding,
            using_arg_candidate, strict_postprocessing,
    ):
        self.grammar = grammar
        self.compiler = compiler
        self.model = learning.load_model(model_path, num_tokens=len(grammar.lf_tokenizer))
        self.model.to(device)
        self.context = context
        self.num_beams = num_beams
        self.generation_max_length = generation_max_length
        self.constrained_decoding = constrained_decoding
        self.using_arg_candidate = using_arg_candidate
        self.strict_postprocessing = strict_postprocessing

        self.collate = make_collate(decoder_start_token_id=grammar.model_config.decoder_start_token_id,
                                    pad_token_id=grammar.lf_tokenizer.pad_token_id,)

    @interface.implement
    def process(self, dataset_part):
        batch = self.collate(dataset_part)
        batch_size = len(dataset_part)

        if self.constrained_decoding:
            logits_processor = learning.get_logits_processor(
                self.grammar, batch_size, self.num_beams, renormalizing=False,
                utterance_token_ids=batch['ws_utterance_token_ids'])
        else:
            logits_processor = None

        token_id_seqs = learning.generate_token_id_seqs(
            grammar=self.grammar,
            model=self.model,
            utterance_token_ids=batch['utterance_token_ids'].to(self.model.device),
            max_length=self.generation_max_length,
            num_beams=self.num_beams,
            num_return_sequences=self.num_beams,
            logits_processor=logits_processor,
            # **generation_kwargs
        )

        last_states = learning.token_id_seqs_to_last_states(
            self.grammar, token_id_seqs,
            ignoring_parsing_errors=True,
            verifying=False,
            utterance_token_id_seqs=(batch['utterance_token_ids'].tolist() if self.using_arg_candidate else None),
            num_return_sequences=self.num_beams
        )

        programs = learning.last_states_to_programs(
            self.grammar, self.compiler, last_states, tolerant=True, ignoring_compilation_errors=True)

        predictions = learning.programs_to_predictions(self.context, programs, strict_postprocessing=self.strict_postprocessing)

        action_id_seq_groups = get_consistent_action_id_seq_groups(
            action_id_seqs=token_id_seqs,
            predictions=predictions,
            answers=batch['answer'],
            num_return_sequences=self.num_beams)

        return action_id_seq_groups


def optimize_with_weaksup(
        pretrained_model_name_or_path,
        grammar,
        compiler,
        device,
        devices,
        logger,
        encoded_weaksup_train_set,
        encoded_val_set,
        train_batch_size,
        val_batch_size,
        learning_rate,
        adam_epsilon,
        weight_decay,
        num_train_epochs,
        using_scheduler,
        num_warmup_epochs,
        max_grad_norm,
        patience,
        constrained_decoding,
        using_arg_candidate,
        model_learning_dir_path,
        num_prediction_beams,
        generation_max_length,
):

    last_dir_path = learning.get_last_dir_path(model_learning_dir_path)
    filesys.mkloc_unless_exist(last_dir_path)
    best_dir_path = learning.get_best_dir_path(model_learning_dir_path)
    filesys.mkloc_unless_exist(best_dir_path)

    model = learning.load_model(
        pretrained_model_name_or_path,
        num_tokens=len(grammar.lf_tokenizer))
    model.to(device)

    train_data_loader = make_data_loader(
        encoded_dataset=encoded_weaksup_train_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=train_batch_size,
        shuffle=True)
    val_data_loader = make_data_loader(
        encoded_dataset=encoded_val_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=val_batch_size,
        shuffle=False)

    param_groups = learning.get_param_groups(model, learning_rate=learning_rate, weight_decay=weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, eps=adam_epsilon)

    scheduler = learning.make_scheduler(
        optimizer=optimizer,
        using_scheduler=using_scheduler,
        train_data_loader=train_data_loader,
        num_train_epochs=num_train_epochs,
        num_warmup_epochs=num_warmup_epochs,
    )

    status = get_init_status(measures=optim_measures, update_unit='epoch')


    raise NotImplementedError
