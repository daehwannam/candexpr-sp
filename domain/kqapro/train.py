
import os
import torch
# from tqdm import tqdm

from configuration import config

from . import learning
from .data_read import make_data_loader
from .execution import postprocess_prediction

from dhnamlib.pylib import filesys
from dhnamlib.pylib.context import replace_dir, copy_dir
# from dhnamlib.pylib.iteration import apply_recursively
from dhnamlib.pylib.iteration import pairs2dicts, not_none_valued_pairs, xtqdm
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
        train_batch_size=config.ph,
        val_batch_size=config.ph,
        learning_rate=config.ph,
        adam_epsilon=config.ph,
        weight_decay=config.ph,
        num_train_epochs=config.ph,
        num_warmup_epochs=config.ph,
        max_grad_norm=config.ph,
        # softmax_masking=config.ph,
        softmax_masking=False,
        model_dir_path=None,
        restarting=False,
        context=config.ph,
        num_prediction_beams=config.ph,
        generation_max_length=config.ph,
):
    if restarting:
        assert learning.is_finetuned(pretrained_model_name_or_path)

        if model_dir_path is None:
            model_dir_path = filesys.get_parent_path(pretrained_model_name_or_path)

    last_dir_path = os.path.join(model_dir_path, 'last')
    filesys.mkloc_unless_exist(last_dir_path)
    best_dir_path = os.path.join(model_dir_path, 'best')
    filesys.mkloc_unless_exist(best_dir_path)

    model = learning.load_model(
        pretrained_model_name_or_path,
        num_tokens=len(grammar.lf_tokenizer))
    model.to(device)

    train_data_loader = make_data_loader(
        encoded_dataset=encoded_train_set,
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
    num_training_steps = len(train_data_loader) * num_train_epochs
    num_warmup_steps = len(train_data_loader) * num_warmup_epochs

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
        # if config.debug:
        #     batch_idx = -1
        loss = torch.tensor(0.)
        for batch in xtqdm(train_data_loader, desc_fn=lambda: 'loss: {:7.4f}'.format(loss.item())):
            # if config.debug:
            #     batch_idx += 1
            #     if batch_idx >= 100:
            #         break

            # TODO
            # - Use `model.config.decoder_start_token_id` as the first id of sequences.
            # - decoder_start_token_id -> bos_token_id -> others ...
            batched_input = dict(
                input_ids=batch['utterance_token_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device))

            optimizer.zero_grad()
            batched_output = model(**batched_input)

            logits = batched_output['logits']
            labels = batch['labels'].to(device)
            # if softmax_masking:
            #     softmax_mask = batch['softmax_mask'].to(device)
            # nll_mask = batch['nll_mask'].to(device)
            if softmax_masking:
                softmax_mask, nll_mask = learning.labels_to_masks(grammar, labels)
            else:
                softmax_mask = None
                nll_mask = learning.labels_to_nll_mask(grammar, labels)
            loss = learning.compute_loss(grammar, logits, labels,
                                         softmax_mask=softmax_mask,
                                         nll_mask=nll_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

        model.eval()
        # from dhnamlib.pylib.cProfiling import run_context
        # run_context('''validation = validate( grammar=grammar, compiler=compiler, model=model, context=context, data_loader=val_data_loader, batch_size=val_batch_size, num_beams=num_prediction_beams, generation_max_length=generation_max_length)''', sort='cumtime')
        validation = validate(
            grammar=grammar,
            compiler=compiler,
            model=model,
            context=context,
            data_loader=val_data_loader,
            batch_size=val_batch_size,
            num_beams=num_prediction_beams,
            generation_max_length=generation_max_length)

        performance = validation['performance']

        status.update(
            last_performance=performance,
            last_epoch=epoch)
        status.history.append(
            dict(epoch=epoch,
                 performance=performance))

        logger.info(f'Epoch: {epoch} / Performance: {str(performance)}')

        # breakpoint()
        updating_best = is_better_performance(performance, status.best_performance, measures)

        if updating_best:
            status.update(
                best_performance=performance,
                best_epoch=epoch)
            logger.info('Best model is updated')

        with replace_dir(last_dir_path) as temp_last_dir_path:
            # save
            learning.save_status(status, temp_last_dir_path)
            learning.save_optimizer(optimizer, temp_last_dir_path)
            learning.save_scheduler(scheduler, temp_last_dir_path)
            learning.save_model(model, temp_last_dir_path)
            learning.save_analysis(validation['analysis'], temp_last_dir_path)

        if updating_best:
            copy_dir(last_dir_path, best_dir_path, replacing=True)


def validate(
        grammar,
        compiler,
        model,
        context,
        data_loader,
        batch_size,
        num_beams,
        generation_max_length,
        analyzing=True,
        softmax_masking=False,
):
    if softmax_masking:
        generation_kwargs = dict(
            logits_processor=learning.get_logits_processor(grammar, batch_size, num_beams))
    else:
        generation_kwargs = dict(
            prefix_allowed_tokens_fn=grammar.make_prefix_allowed_tokens_fn(batch_size, num_beams))

    all_predictions = []
    all_answers = []

    if analyzing:
        all_utterances = []
        all_predicted_last_states = []
        all_answer_last_states = []

    # if config.debug:
    #     batch_idx = -1

    for batch in xtqdm(data_loader):
        # if config.debug:
        #     batch_idx += 1
        #     if batch_idx >= 5:
        #         break

        last_states = learning.parse(
            grammar=grammar,
            model=model,
            utterance_ids=batch['utterance_token_ids'].to(model.device),
            max_length=generation_max_length,
            num_beams=num_beams,
            **generation_kwargs
        )
        programs = learning.last_states_to_programs(grammar, compiler, last_states, tolerant=True)
        predictions = learning.programs_to_predictions(context, programs)
        answers = batch['answer']

        all_predictions.extend(predictions)
        all_answers.extend(answers)

        if analyzing:
            utterances = grammar.utterance_tokenizer.batch_decode(
                batch['utterance_token_ids'], skip_special_tokens=True)
            if 'labels' in batch:
                answer_last_states = tuple(learning.token_id_seq_to_last_state(grammar, token_id_seq)
                                           for token_id_seq in batch['labels'].tolist())
                all_answer_last_states.extend(answer_last_states)
            all_utterances.extend(utterances)
            all_predicted_last_states.extend(last_states)

    accuracy = compute_accuracy(all_predictions, all_answers)
    performance = get_performance(accuracy=accuracy)

    if analyzing:
        def get_action_seq(last_state):
            if grammar.is_invalid_state(last_state):
                return None
            else:
                if last_state.tree.is_closed_root():
                    return list(map(repr, last_state.tree.get_values()))
                else:
                    return None

        def get_tree_repr(last_state):
            if grammar.is_invalid_state(last_state):
                return None
            else:
                return repr(last_state.tree)

        def get_expr_str(last_state, expr_key=None):
            if grammar.is_invalid_state(last_state):
                return None
            else:
                if last_state.tree.is_closed_root():
                    return last_state.tree.get_expr_str(expr_key=expr_key)
                else:
                    return None

        def analyze_program(last_states):
            program_analysis = list(pairs2dicts(
                action_seq=list(map(get_action_seq, last_states)),
                tree=list(map(get_tree_repr, last_states)),
                expr=list(map(get_expr_str, last_states)),
                visual_expr=list(map(lambda last_state: get_expr_str(last_state, expr_key='visual'), last_states)),
            ))
            return program_analysis

        analysis = list(pairs2dicts(not_none_valued_pairs(
            utterance=all_utterances,
            answer=all_answers,
            prediction=all_predictions,
            predicted_program=analyze_program(all_predicted_last_states),
            answer_program=(analyze_program(all_answer_last_states)
                            if len(all_answer_last_states) > 0 else None))))

    validation = dict(
        performance=performance,
        analysis=analysis
    )

    return validation


def compute_accuracy(predictions, answers):
    assert len(predictions) == len(answers)
    num_examples = len(predictions)

    num_correct = sum(
        int(prediction == answer)
        for prediction, answer in zip(predictions, answers))

    return num_correct / num_examples


if __name__ == '__main__':
    train(model_dir_path=config.model_dir_path)

    # from dhnamlib.pylib.cProfiling import run_context
    # tuple(config.items(lazy=False))
    # run_context('train(model_dir_path=config.model_dir_path)', sort='cumtime')
