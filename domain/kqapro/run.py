
import os
import torch
# from tqdm import tqdm
import transformers

from configuration import config, save_config_info

from . import learning
from .data_read import make_data_loader
from .execution import postprocess_prediction

from kqapro.evaluate import whether_equal

from dhnamlib.pylib import filesys
# from dhnamlib.pylib.iteration import apply_recursively
from dhnamlib.pylib.iteration import pairs2dicts, not_none_valued_pairs
from dhnamlib.pylib.structure import AttrDict

from dhnamlib.pylib.torchlib.optimization import get_linear_schedule_with_warmup
from dhnamlib.pylib.torchlib.stat import get_performance, get_measure, is_better_performance


@config
def run_train(
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
        using_scheduler=config.ph,
        num_warmup_epochs=config.ph,
        max_grad_norm=config.ph,
        softmax_masking=config.ph,
        # softmax_masking=False,
        constrained_decoding=config.ph,
        model_learning_dir_path=config.ph,
        restarting=False,
        context=config.ph,
        num_prediction_beams=config.ph,
        generation_max_length=config.ph,
):
    if restarting:
        assert learning.is_finetuned(pretrained_model_name_or_path)

        if model_learning_dir_path is None:
            model_learning_dir_path = filesys.get_parent_path(pretrained_model_name_or_path)
    else:
        assert model_learning_dir_path is not None

    save_config_info(model_learning_dir_path)

    last_dir_path = learning.get_last_dir_path(model_learning_dir_path)
    filesys.mkloc_unless_exist(last_dir_path)
    best_dir_path = learning.get_best_dir_path(model_learning_dir_path)
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

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, eps=adam_epsilon)

    if using_scheduler:
        num_training_steps = len(train_data_loader) * num_train_epochs
        num_warmup_steps = len(train_data_loader) * num_warmup_epochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    else:
        scheduler = transformers.get_constant_schedule(optimizer)

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
        #     if epoch > 3:
        #         break

        # if config.debug:
        #     batch_idx = -1
        loss = torch.tensor(0.)
        for batch in config.xtqdm(train_data_loader, desc_fn=lambda: 'loss: {:7.4f}'.format(loss.item())):
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
            generation_max_length=generation_max_length,
            softmax_masking=softmax_masking,
            constrained_decoding=constrained_decoding,
            evaluating=True)

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

        with filesys.replace_dir(last_dir_path) as temp_last_dir_path:
            # save
            learning.save_status(status, temp_last_dir_path)
            learning.save_performance(performance, temp_last_dir_path)
            learning.save_optimizer(optimizer, temp_last_dir_path)
            learning.save_scheduler(scheduler, temp_last_dir_path)
            learning.save_model(model, temp_last_dir_path)
            learning.save_analysis(validation['analysis'], temp_last_dir_path)

        if updating_best:
            filesys.copy_dir(last_dir_path, best_dir_path, replacing=True)

        logger.info(f'Results are saved in "{model_learning_dir_path}"')


@config
def run_test(
        *,
        model_learning_dir_path=config.ph,
        test_dir_path=config.ph,
        grammar=config.ph,
        compiler=config.ph,
        device=config.ph,
        logger=config.ph,
        encoded_test_set,
        test_batch_size=config.ph,
        softmax_masking=config.ph,
        constrained_decoding=config.ph,
        context=config.ph,
        num_prediction_beams=config.ph,
        generation_max_length=config.ph,
        evaluating,
):
    model = learning.load_model(
        learning.get_best_dir_path(model_learning_dir_path),
        num_tokens=len(grammar.lf_tokenizer))
    model.to(device)

    test_data_loader = make_data_loader(
        encoded_dataset=encoded_test_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=test_batch_size,
        shuffle=False)

    model.eval()
    validation = validate(
        grammar=grammar,
        compiler=compiler,
        model=model,
        context=context,
        data_loader=test_data_loader,
        batch_size=test_batch_size,
        num_beams=num_prediction_beams,
        generation_max_length=generation_max_length,
        evaluating=evaluating,
        softmax_masking=softmax_masking,
        constrained_decoding=constrained_decoding)

    if evaluating:
        logger.info('Performance: {}'.format(validation['performance']))

    filesys.mkloc_unless_exist(test_dir_path)
    save_config_info(test_dir_path)

    learning.save_analysis(validation['analysis'], test_dir_path)
    learning.save_predictions(validation['predictions'], test_dir_path)
    if evaluating:
        learning.save_performance(validation['performance'], test_dir_path)

    logger.info(f'Results are saved in "{test_dir_path}"')


def validate(
        *,
        grammar,
        compiler,
        model,
        context,
        data_loader,
        batch_size,
        num_beams,
        generation_max_length,
        analyzing=True,
        softmax_masking,
        constrained_decoding,
        evaluating,
):
    assert not model.training

    assert constrained_decoding or not softmax_masking
    if constrained_decoding:
        logits_processor = learning.get_logits_processor(
            grammar, batch_size, num_beams, renormalizing=softmax_masking)
    else:
        logits_processor = None

    # if softmax_masking:
    #     generation_kwargs = dict(
    #         logits_processor=learning.get_rescaled_logits_processor(grammar, batch_size, num_beams))
    # else:
    #     generation_kwargs = dict(
    #         prefix_allowed_tokens_fn=learning.make_prefix_allowed_tokens_fn(grammar, batch_size, num_beams))

    all_predictions = []
    if evaluating:
        all_answers = []

    if analyzing:
        all_utterances = []
        all_predicted_token_id_seqs = []
        all_predicted_last_states = []
        if evaluating:
            all_answer_last_states = []

    if evaluating:
        realtime_num_correct = 0

        def update_realtime_accuracy():
            nonlocal realtime_num_correct
            realtime_num_correct += compute_num_correct(predictions, answers)
            assert len(all_predictions) == len(all_answers)
            return realtime_num_correct / len(all_predictions)

        xtqdm_kwargs = dict(
            desc='accuracy: none',
            desc_fn=lambda: 'accuracy: {:4.1f}'.format(update_realtime_accuracy() * 100))
    else:
        xtqdm_kwargs = dict()

    # if config.debug:
    #     batch_idx = -1

    for batch in config.xtqdm(data_loader, **xtqdm_kwargs):
        # if config.debug:
        #     batch_idx += 1
        #     # if batch_idx >= 10:
        #     #     break
        #     # if batch_idx < 1000:
        #     if batch_idx < 500:
        #         def update_realtime_accuracy():
        #             return 0
        #         continue
        #     utterances = grammar.utterance_tokenizer.batch_decode(batch['utterance_token_ids'], skip_special_tokens=True)
        #     # assert len(utterances) == 1
        #     test_utterance = "How many contemporary folk music groups are related to famous Taj Mahal (the one whose ISNI is 0000 0001 1598 8398 or KT Tunstall?"
        #     if test_utterance in utterances:
        #         breakpoint()

        token_id_seqs = learning.generate_token_id_seqs(
            grammar=grammar,
            model=model,
            utterance_token_ids=batch['utterance_token_ids'].to(model.device),
            max_length=generation_max_length,
            num_beams=num_beams,
            logits_processor=logits_processor,
            # **generation_kwargs
        )
        last_states = learning.token_id_seqs_to_last_states(grammar, token_id_seqs)
        programs = learning.last_states_to_programs(grammar, compiler, last_states, tolerant=True)

        predictions = learning.programs_to_predictions(context, programs)
        all_predictions.extend(predictions)

        if evaluating:
            assert 'answer' in batch
            answers = batch['answer']
            all_answers.extend(answers)

        if analyzing:
            utterances = grammar.utterance_tokenizer.batch_decode(
                batch['utterance_token_ids'], skip_special_tokens=True)

            if evaluating:
                assert 'labels' in batch
                answer_last_states = tuple(learning.token_id_seq_to_last_state(grammar, token_id_seq)
                                           for token_id_seq in batch['labels'].tolist())
                all_answer_last_states.extend(answer_last_states)
            all_utterances.extend(utterances)
            all_predicted_token_id_seqs.extend(token_id_seqs)
            all_predicted_last_states.extend(last_states)

    if evaluating:
        assert len(all_predictions) == len(all_answers) == len(all_answer_last_states)
        accuracy = compute_accuracy(all_predictions, all_answers)
        performance = get_performance(accuracy=accuracy)
    else:
        performance = None

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

        def analyze_program(last_states, token_id_seqs=None):
            program_analysis = list(pairs2dicts(not_none_valued_pairs(
                tokens=list(map(grammar.lf_tokenizer.convert_ids_to_tokens, token_id_seqs)) if token_id_seqs is not None else None,
                action_seq=list(map(get_action_seq, last_states)),
                tree=list(map(get_tree_repr, last_states)),
                expr=list(map(get_expr_str, last_states)),
                visual_expr=list(map(lambda last_state: get_expr_str(last_state, expr_key='visual'), last_states)),
            )))
            return program_analysis

        # breakpoint()

        analysis = list(pairs2dicts(not_none_valued_pairs(
            example_idx=tuple(range(len(all_utterances))),
            utterance=all_utterances,
            answer=all_answers if evaluating else None,
            prediction=all_predictions,
            predicted_program=analyze_program(all_predicted_last_states, all_predicted_token_id_seqs),
            answer_program=(analyze_program(all_answer_last_states) if evaluating else None))))

    validation = dict(not_none_valued_pairs(
        performance=performance,
        analysis=analysis,
        predictions=all_predictions))

    return validation


def compute_num_correct(predictions, answers):
    assert len(predictions) == len(answers)

    num_correct = sum(
        int(whether_equal(answer=answer, prediction=prediction))
        for prediction, answer in zip(predictions, answers))

    return num_correct


def compute_accuracy(predictions, answers):
    assert len(predictions) == len(answers)
    num_examples = len(predictions)

    accuracy = compute_num_correct(predictions, answers) / num_examples
    return accuracy


if __name__ == '__main__':
    if config.run_mode == 'train':
        run_train()
        # from dhnamlib.pylib.cProfiling import run_context
        # run_context('run_train(model_learning_dir_path=config.model_learning_dir_path)', sort='cumtime')
    elif config.run_mode == 'test-on-val-set':
        run_test(encoded_test_set=config.encoded_val_set, evaluating=True)
        # from dhnamlib.pylib.cProfiling import run_context
        # run_context('run_test(encoded_test_set=config.encoded_val_set, evaluating=True)', sort='cumtime')
    elif config.run_mode == 'test-on-test-set':
        run_test(encoded_test_set=config.encoded_test_set, evaluating=False)
    else:
        raise Exception(f'Unknown execution type "{config.run_mode}"')

    # from dhnamlib.pylib.cProfiling import run_context
    # tuple(config.items(lazy=False))
    # run_context('run_train(model_learning_dir_path=config.model_learning_dir_path)', sort='cumtime')
