
import os
import torch
from tqdm import tqdm

from configuration import config

from . import learning
from .data_read import make_data_loader
from .execution import postprocess_prediction

from dhnamlib.pylib import filesys
from dhnamlib.pylib.context import block
# from dhnamlib.pylib.iteration import apply_recursively
from dhnamlib.pylib.iteration import pairs2dicts, filter_dict_values, is_not_none
from dhnamlib.pylib.structure import AttrDict

from dhnamlib.pylib.torchlib.optimization import get_linear_schedule_with_warmup
from dhnamlib.pylib.torchlib.stat import get_performance, get_measure, is_better_performance


@config
def train(
        pretrained_model_name_or_path=config.ph,
        grammar=config.ph,
        compiler=config.ph,
        device=config.ph,
        logger=config.ph,
        encoded_train_set=config.ph,
        encoded_val_set=config.ph,
        batch_size=config.ph,
        weight_decay=config.ph,
        learning_rate=config.ph,
        num_train_epochs=config.ph,
        num_warmup_steps=config.ph,
        adam_epsilon=config.ph,
        output_dir_path=None,
        restarting=False,
        context=config.ph,
        num_beams=config.ph,
        generation_max_length=config.ph,
):
    if restarting:
        assert learning.is_finetuned(pretrained_model_name_or_path)

        if output_dir_path is None:
            output_dir_path = filesys.get_parent_path(pretrained_model_name_or_path)

    last_dir_path = os.path.join(output_dir_path, 'last')
    filesys.mkloc_unless_exist(last_dir_path)
    best_dir_path = os.path.join(output_dir_path, 'best')
    filesys.mkloc_unless_exist(best_dir_path)

    model = learning.load_model(
        pretrained_model_name_or_path,
        num_tokens=len(grammar.lf_tokenizer))
    model.to(device)

    train_data_loader = make_data_loader(
        encoded_dataset=encoded_train_set,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=batch_size,
        shuffle=True)
    val_batch_size = batch_size * 4
    val_data_loader = make_data_loader(
        encoded_dataset=encoded_val_set,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=val_batch_size,
        shuffle=False)

    param_groups = learning.get_param_groups(model, learning_rate=learning_rate, weight_decay=weight_decay)
    num_training_steps = len(train_data_loader) * num_train_epochs

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    if restarting:
        # TODO: what do optimizer and scheduler save?
        learning.load_and_update_optimizer(optimizer, pretrained_model_name_or_path)
        learning.load_and_update_scheduler(scheduler, pretrained_model_name_or_path)

        status = AttrDict(learning.load_status(pretrained_model_name_or_path))
    else:
        _last_performance = get_performance(accuracy=float('-inf'))
        status = AttrDict(
            last_epoch=0,
            best_epoch=0,
            last_performance=_last_performance,
            best_performance=_last_performance,
            history=[])
    measures = [get_measure('accuracy', True)]

    for epoch in range(status['last_epoch'] + 1, num_train_epochs + 1):
        logger.info(f'Epoch {epoch} starts')
        model.train()
        for batch in tqdm(train_data_loader):
            batched_input = dict(
                input_ids=batch['utterance_token_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device),
                labels=batch['labels'].to(device))

            optimizer.zero_grad()
            batched_output = model(**batched_input)

            raise NotImplementedError('apply masked_softmax')

            with block:
                logits = batched_output['logits']
            loss = ...
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        validation_result = validate(
            grammar=grammar,
            compiler=compiler,
            model=model,
            context=context,
            data_loader=val_data_loader,
            device=device,
            batch_size=val_batch_size,
            num_beams=num_beams,
            generation_max_length=generation_max_length)

        performance = validation_result['performance']

        status.update(
            last_performance=performance,
            last_epoch=epoch)
        status.history.append(
            dict(epoch=epoch,
                 performance=performance))

        updating_best = is_better_performance(performance, status.best_performance, measures)

        if updating_best:
            status.update(
                best_performance=performance,
                best_epoch=epoch)

        with block:
            # save
            learning.save_status(status, last_dir_path)
            learning.save_optimizer(optimizer, last_dir_path)
            learning.save_scheduler(scheduler, last_dir_path)
            learning.save_model(model, last_dir_path)
            raise NotImplementedError('save validation detailed result')

            if updating_best:
                raise NotImplementedError('save best model (copy from the last)')

def validate(
        grammar,
        compiler,
        model,
        context,
        data_loader,
        device,
        batch_size,
        num_beams,
        generation_max_length,
        analyzing=True,
):
    logits_processor = learning.get_logits_processor(grammar, batch_size, num_beams)

    all_predictions = []
    all_answers = []

    if analyzing:
        all_utterances = []
        all_predicted_last_states = []
        all_answer_last_states = []

    for batch in tqdm(data_loader):
        batched_output = model.generate(
            input_ids=batch['utterance_token_ids'].to(device),
            max_length=generation_max_length,
            logits_processor=logits_processor,
            num_beams=num_beams,
        )
        predicted_last_states = tuple(learning.token_id_seq_to_last_state(grammar, token_id_seq)
                                      for token_id_seq in batched_output)
        predicted_programs = tuple(compiler.compile_tree(last_state.tree)
                                   for last_state in predicted_last_states)
        predictions = tuple(postprocess_prediction(program(context))
                            for program in predicted_programs)
        answers = batch['answer']

        all_predictions.extend(predictions)
        all_answers.extend(answers)

        if analyzing:
            utterances = grammar.utteance_tokenizer.batch_decode(
                batch['utterance_token_ids'], skip_special_tokens=True)
            if 'labels' in batch['labels']:
                answer_last_states = tuple(learning.token_id_seq_to_last_state(grammar, batch['labels']))
                all_answer_last_states.extend(answer_last_states)
            all_utterances.extend(utterances)
            all_predicted_last_states.extend(predicted_last_states)

    accuracy = compute_accuracy(all_predictions, all_answers)
    performance = get_performance(accuracy=accuracy)

    if analyzing:
        def analyze_program(last_states):
            program_analysis = list(pairs2dicts(
                action_seq=[list(map(repr, last_state.tree.get_values())) for last_state in last_states],
                tree=[repr(last_state.tree) for last_state in last_states],
                expr=[last_state.tree.get_expr_str() for last_state in last_states],
            ))
            return program_analysis

        analysis = list(pairs2dicts(filter_dict_values(
            is_not_none,
            dict(
                utterance=all_utterances,
                answer=all_answers,
                prediction=all_predictions,
                predicted_program=analyze_program(all_predicted_last_states),
                answer_program=(analyze_program(all_answer_last_states)
                                if len(all_answer_last_states) > 0 else None)))))

    return dict(
        performance=performance,
        analysis=analysis
    )


def compute_accuracy(predictions, answers):
    assert len(predictions) == len(answers)
    num_examples = len(predictions)

    num_correct = sum(
        int(prediction == answer)
        for prediction, answer in zip(predictions, answers))

    return num_correct / num_examples


if __name__ == '__main__':
    train()

