
import os
# import warnings

import torch
# from tqdm import tqdm

# from functools import lru_cache

from configuration import config, save_config_info

from . import learning
from .data_read import make_data_loader
from .validation import validate
from .learning import optim_measures, search_measures
from .weaksup import search_to_collect


from dhnamlib.pylib import filesys
# from dhnamlib.pylib.iteration import apply_recursively
# from dhnamlib.pylib.structure import AttrDict
from dhnamlib.pylib.context import skippable, skip_if_possible

from dhnamlib.pylib.mllib.learning import get_init_status, update_status
# from dhnamlib.pylib.mllib.learning import get_performance


# wlmp = within_local_main_process
save_config_info_wlmp = config.accelerator.within_local_main_process(save_config_info)
mkloc_unless_exist_wlmp = config.accelerator.within_local_main_process(filesys.mkloc_unless_exist)
copy_dir_wlmp = config.accelerator.within_local_main_process(filesys.copy_dir)
copy_matched_wlmp = config.accelerator.within_local_main_process(filesys.copy_matched)
replace_dir_wlmp = filesys.replace_dir if config.accelerator.is_local_main_process else skippable
skip_if_not_wlmp = skip_if_possible


@config
def run_train(
        pretrained_model_name_or_path=config.ph,
        grammar=config.ph,
        compiler=config.ph,
        # device=config.ph,
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
        patience=config.ph,
        softmax_masking=config.ph,
        constrained_decoding=config.ph,
        using_arg_candidate=config.ph,
        model_learning_dir_path=config.ph,
        restarting=False,
        context=config.ph,
        num_prediction_beams=config.ph,
        generation_max_length=config.ph,
        saving_optimizer=config.ph,
):
    if restarting:
        assert learning.is_finetuned(pretrained_model_name_or_path)

        if model_learning_dir_path is None:
            model_learning_dir_path = filesys.get_parent_path(pretrained_model_name_or_path)
    else:
        assert model_learning_dir_path is not None

    save_config_info_wlmp(model_learning_dir_path)

    last_dir_path = learning.get_last_dir_path(model_learning_dir_path)
    mkloc_unless_exist_wlmp(last_dir_path)
    best_dir_path = learning.get_best_dir_path(model_learning_dir_path)
    mkloc_unless_exist_wlmp(best_dir_path)

    model = learning.load_model(
        pretrained_model_name_or_path,
        num_tokens=len(grammar.lf_tokenizer))
    # model.to(device)
    model.to(config.accelerator.device)

    train_data_loader = make_data_loader(
        encoded_dataset=encoded_train_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=train_batch_size,
        shuffle=True,
    )
    val_data_loader = make_data_loader(
        encoded_dataset=encoded_val_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=val_batch_size,
        shuffle=False,
    )

    param_groups = learning.get_param_groups(model, learning_rate=learning_rate, weight_decay=weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, eps=adam_epsilon)

    scheduler = learning.make_scheduler(
        optimizer=optimizer,
        using_scheduler=using_scheduler,
        train_data_loader=train_data_loader,
        num_train_epochs=num_train_epochs,
        num_warmup_epochs=num_warmup_epochs,
    )

    if restarting:
        # Question: what do optimizer and scheduler save?
        learning.load_and_update_optimizer(optimizer, pretrained_model_name_or_path)
        learning.load_and_update_scheduler(scheduler, pretrained_model_name_or_path)

        status = dict(learning.load_status(pretrained_model_name_or_path))
    else:
        status = get_init_status(measures=optim_measures, update_unit='epoch')

    # The model and optimizer should be passed together to the prepare method
    # https://huggingface.co/docs/accelerate/quicktour#enable-distributed-training-in-your-script
    model, train_data_loader, val_data_loader, optimizer, scheduler = config.accelerator.prepare(
        model, train_data_loader, optimizer, scheduler)

    val_data_loader = config.accelerator.prepare_val_data_loader(val_data_loader)

    remaining_patience = patience

    # TODO: use accelerate.local_sgd.LocalSGD
    # https://huggingface.co/docs/accelerate/usage_guides/local_sgd

    for epoch in range(status['last_update_num'] + 1, num_train_epochs + 1):
        logger.info(f'Epoch {epoch} starts')
        model.train()

        # debug_batch_idx = -1
        loss = torch.tensor(0.)
        # for batch in config.xtqdm(train_data_loader, desc_fn=lambda: 'loss: {:7.4f}'.format(loss.item())):
        for batch in config.utqdm(train_data_loader, unit='loss', update_fn=lambda: loss.item(), repr_format='{:7.4f}'):
            # - `model.config.decoder_start_token_id` is the first id of output sequences.
            # - the order or decoder output tokens in a sequence: decoder_start_token_id, bos_token_id, others ...
            batched_input = dict(
                input_ids=batch['utterance_token_ids'].to(config.accelerator.device),
                attention_mask=batch['attention_mask'].to(config.accelerator.device),
                decoder_input_ids=batch['decoder_input_ids'].to(config.accelerator.device))

            optimizer.zero_grad()
            batched_output = model(**batched_input)

            logits = batched_output['logits']
            # labels = batch['labels'].to(device)
            labels = batch['labels'].to(config.accelerator.device)

            # if softmax_masking:
            #     softmax_mask = batch['softmax_mask'].to(device)
            # nll_mask = batch['nll_mask'].to(device)
            if softmax_masking:
                softmax_mask, nll_mask = learning.labels_to_masks(grammar, labels, batch['utterance_token_ids'])
            else:
                softmax_mask = None
                nll_mask = learning.labels_to_nll_mask(grammar, labels)
            loss = learning.compute_loss(grammar, logits, labels,
                                         softmax_mask=softmax_mask,
                                         nll_mask=nll_mask)

            config.accelerator.backward(loss)
            if config.accelerator.sync_gradients:
                config.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

        config.accelerator.wait_for_everyone()

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
            using_arg_candidate=using_arg_candidate,
            evaluating=True)

        performance = validation['performance']

        logger.info(f'Epoch: {epoch} / Performance: {str(performance)}')

        updating_best = update_status(status, performance=performance)

        with replace_dir_wlmp(last_dir_path) as temp_last_dir_path:
            skip_if_not_wlmp(temp_last_dir_path)

            # save
            learning.save_status(status, temp_last_dir_path)
            learning.save_performance(performance, temp_last_dir_path)
            # if saving_optimizer:
            #     learning.save_optimizer(optimizer, temp_last_dir_path)
            # learning.save_scheduler(scheduler, temp_last_dir_path)
            learning.save_model(model, temp_last_dir_path)
            learning.save_analysis(validation['analysis'], temp_last_dir_path)

        if updating_best:
            copy_dir_wlmp(last_dir_path, best_dir_path, replacing=True)
            logger.info('Best model is updated')
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience < 0:
                break

        logger.info(f'Results are saved in "{model_learning_dir_path}"')


@config
def run_train_for_multiple_decoding_strategies(
        pretrained_model_name_or_path=config.ph,
        grammar=config.ph,
        compiler=config.ph,
        # device=config.ph,
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
        patience=config.ph,
        softmax_masking=config.ph,
        # constrained_decoding=config.ph,
        # using_arg_candidate=config.ph,
        model_learning_dir_path=config.ph,
        restarting=False,
        context=config.ph,
        num_prediction_beams=config.ph,
        generation_max_length=config.ph,
        saving_optimizer=config.ph,

        # Argument for multiple decoding strategies
        decoding_strategy_configs=config.ph,

        # Argument for limited size of training data
        num_epoch_repeats=config.ph(1),
):
    assert model_learning_dir_path is not None
    assert patience == float('inf'), 'the feature of patience is not implemented'

    save_config_info_wlmp(model_learning_dir_path)

    last_common_dir = learning.get_last_dir_path(model_learning_dir_path, 'last:common')
    mkloc_unless_exist_wlmp(last_common_dir)

    def get_last_dir_path(decoding_strategy_name):
        last_dir_path = learning.get_last_dir_path(
            model_learning_dir_path, f'{decoding_strategy_name}:last')
        mkloc_unless_exist_wlmp(last_dir_path)
        return last_dir_path

    def get_best_dir_path(decoding_strategy_name):
        best_dir_path = learning.get_best_dir_path(
            model_learning_dir_path, f'{decoding_strategy_name}:best')
        mkloc_unless_exist_wlmp(best_dir_path)
        return best_dir_path

    def get_best_result_dir_path(decoding_strategy_name):
        best_result_dir_path = learning.get_best_dir_path(
            model_learning_dir_path, f'{decoding_strategy_name}:best-result')
        mkloc_unless_exist_wlmp(best_result_dir_path)
        return best_result_dir_path

    model = learning.load_model(
        pretrained_model_name_or_path,
        num_tokens=len(grammar.lf_tokenizer))
    model.to(config.accelerator.device)

    train_data_loader = make_data_loader(
        encoded_dataset=encoded_train_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=train_batch_size,
        shuffle=True,
        num_epoch_repeats=num_epoch_repeats)
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

    assert not restarting, 'The feature of restarting is not implemented'

    status_dict = dict(
        [decoding_strategy_config.decoding_strategy_name,
         get_init_status(measures=optim_measures, update_unit='epoch')]
        for decoding_strategy_config in decoding_strategy_configs)

    # The model and optimizer should be passed together to the prepare method
    # https://huggingface.co/docs/accelerate/quicktour#enable-distributed-training-in-your-script
    model, train_data_loader, val_data_loader, optimizer, scheduler = config.accelerator.prepare(
        model, train_data_loader, val_data_loader, optimizer, scheduler)

    val_data_loader = config.accelerator.prepare_val_data_loader(val_data_loader)

    assert not softmax_masking, 'The feature of softmax_masking is not implemented'

    for epoch in range(1, num_train_epochs + 1):
        logger.info(f'Epoch {epoch} starts')
        model.train()

        # debug_batch_cnt = -1

        loss = torch.tensor(0.)
        # for batch in config.xtqdm(train_data_loader, desc_fn=lambda: 'loss: {:7.4f}'.format(loss.item())):
        for batch in config.utqdm(train_data_loader, unit='loss', update_fn=lambda: loss.item(), repr_format='{:7.4f}'):
            # debug_batch_cnt += 1
            # if debug_batch_cnt > 100:
            #     break

            batched_input = dict(
                input_ids=batch['utterance_token_ids'].to(config.accelerator.device),
                attention_mask=batch['attention_mask'].to(config.accelerator.device),
                decoder_input_ids=batch['decoder_input_ids'].to(config.accelerator.device))

            optimizer.zero_grad()
            batched_output = model(**batched_input)

            logits = batched_output['logits']
            labels = batch['labels'].to(config.accelerator.device)
            # labels = batch['labels'].to(device)

            softmax_mask = None
            nll_mask = learning.labels_to_nll_mask(grammar, labels)

            loss = learning.compute_loss(grammar, logits, labels,
                                         softmax_mask=softmax_mask,
                                         nll_mask=nll_mask)

            # loss.backward()
            config.accelerator.backward(loss)
            if config.accelerator.sync_gradients:
                config.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

        config.accelerator.wait_for_everyone()

        model.eval()
        last_model_saved = False

        for decoding_strategy_config in decoding_strategy_configs:
            with config.let(decoding_strategy_config.items()):
                logger.info(f'Validation of "{config.decoding_strategy_name}" starts')

                if 'grammar_lazy_obj' in decoding_strategy_config:
                    _grammar = decoding_strategy_config.grammar_lazy_obj.get()
                else:
                    _grammar = grammar

                validation = validate(
                    grammar=_grammar,
                    compiler=compiler,
                    model=model,
                    context=context,
                    data_loader=val_data_loader,
                    batch_size=val_batch_size,
                    num_beams=num_prediction_beams,
                    generation_max_length=generation_max_length,
                    softmax_masking=softmax_masking,
                    constrained_decoding=config.constrained_decoding,
                    using_arg_candidate=config.using_arg_candidate,
                    evaluating=True)

                performance = validation['performance']

                logger.info(f'Decoding strategy: {config.decoding_strategy_name} / Performance: {str(performance)}')

                status = status_dict[config.decoding_strategy_name]

                updating_best = update_status(status, performance=performance)

                if not last_model_saved:
                    # save a model
                    with replace_dir_wlmp(last_common_dir) as temp_last_dir_path:
                        skip_if_not_wlmp(temp_last_dir_path)

                        if saving_optimizer:
                            learning.save_optimizer(optimizer, temp_last_dir_path)
                        learning.save_scheduler(scheduler, temp_last_dir_path)
                        learning.save_model(model, temp_last_dir_path)
                        last_model_saved = True

                # save for a decoding strategy
                strategy_last_dir = get_last_dir_path(config.decoding_strategy_name)
                learning.save_status(status, strategy_last_dir)
                learning.save_performance(performance, strategy_last_dir)
                learning.save_analysis(validation['analysis'], strategy_last_dir)

                if updating_best:
                    strategy_best_dir = get_best_dir_path(config.decoding_strategy_name)
                    copy_dir_wlmp(last_common_dir, strategy_best_dir, replacing=True)
                    strategy_best_result_dir = get_best_result_dir_path(config.decoding_strategy_name)
                    copy_matched_wlmp(os.path.join(strategy_last_dir, '*'), strategy_best_result_dir)

                    logger.info('Best model is updated')

                logger.info(f'Results are saved in "{model_learning_dir_path}"')


@config
def run_test(
        *,
        model_learning_dir_path=config.ph(None),
        model_checkpoint_dir_path=config.ph(None),
        test_dir_path=config.ph,
        grammar=config.ph,
        compiler=config.ph,
        # device=config.ph,
        logger=config.ph,
        encoded_test_set,
        test_batch_size=config.ph,
        softmax_masking=config.ph,
        constrained_decoding=config.ph,
        using_arg_candidate=config.ph,
        context=config.ph,
        num_prediction_beams=config.ph,
        generation_max_length=config.ph,
        evaluating,
        analyzing=True,
        using_oracle=False,
):
    if model_checkpoint_dir_path is None:
        # Check the default checkpoint path
        assert model_learning_dir_path is not None
        model_checkpoint_dir_path = learning.get_best_dir_path(model_learning_dir_path)
        assert os.path.isdir(model_checkpoint_dir_path), "Specify the explicit path to the checkpoint"
    model = learning.load_model(
        model_checkpoint_dir_path,
        num_tokens=len(grammar.lf_tokenizer))
    model.to(config.accelerator.device)

    test_data_loader = make_data_loader(
        encoded_dataset=encoded_test_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=test_batch_size,
        shuffle=False)

    model = config.accelerator.prepare(model)
    test_data_loader = config.accelerator.prepare_val_data_loader(test_data_loader)

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
        analyzing=analyzing,
        evaluating=evaluating,
        softmax_masking=softmax_masking,
        constrained_decoding=constrained_decoding,
        using_arg_candidate=using_arg_candidate,
        using_oracle=using_oracle,
    )

    if evaluating:
        logger.info('Performance: {}'.format(validation['performance']))

    mkloc_unless_exist_wlmp(test_dir_path)
    save_config_info_wlmp(test_dir_path)

    if analyzing:
        learning.save_analysis(validation['analysis'], test_dir_path)
    learning.save_predictions(validation['predictions'], test_dir_path)
    if evaluating:
        learning.save_performance(validation['performance'], test_dir_path)
    learning.save_time_info(validation['time_info'], test_dir_path)

    logger.info(f'Results are saved in "{test_dir_path}"')


# @config
def run_oracle_test(
        *,
        # model_learning_dir_path=config.ph(None),
        # model_checkpoint_dir_path=config.ph(None),
        # test_dir_path=config.ph,
        # grammar=config.ph,
        # compiler=config.ph,
        # device=config.ph,
        # logger=config.ph,
        encoded_test_set,
        # test_batch_size=config.ph,
        # softmax_masking=config.ph,
        # constrained_decoding=config.ph,
        # using_arg_candidate=config.ph,
        # context=config.ph,
        # num_prediction_beams=config.ph,
        # generation_max_length=config.ph,
        evaluating,
        analyzing=False,
):
    run_test(
        # model_learning_dir_path=model_learning_dir_path,
        # model_checkpoint_dir_path=model_checkpoint_dir_path,
        # test_dir_path=test_dir_path,
        # grammar=grammar,
        # compiler=compiler,
        # device=device,
        # logger=logger,
        encoded_test_set=encoded_test_set,
        # test_batch_size=test_batch_size,
        # softmax_masking=softmax_masking,
        # constrained_decoding=constrained_decoding,
        # using_arg_candidate=using_arg_candidate,
        # context=context,
        # num_prediction_beams=num_prediction_beams,
        # generation_max_length=generation_max_length,
        evaluating=evaluating,
        analyzing=analyzing,
        using_oracle=True,
    )


# @config
def run_search_train(
        pretrained_model_name_or_path=config.ph,
        grammar=config.ph,
        compiler=config.ph,
        devices=config.ph,
        logger=config.ph,
        encoded_weaksup_search_set=config.ph,
        encoded_val_set=config.ph,
        train_batch_size=config.ph,
        val_batch_size=config.ph,
        search_batch_size=config.ph,
        learning_rate=config.ph,  # TODO
        adam_epsilon=config.ph,   # TODO
        weight_decay=config.ph,
        num_train_epochs=config.ph,  # TODO
        using_scheduler=config.ph,   # TODO
        num_warmup_epochs=config.ph,  # TODO
        max_grad_norm=config.ph,      # TODO
        patience=config.ph,     # TODO
        softmax_masking=config.ph,
        constrained_decoding=config.ph,
        using_arg_candidate=config.ph,
        model_learning_dir_path=config.ph,
        resuming=False,
        max_num_iterations=None,
        context=config.ph,
        num_search_beams=config.ph,
        generation_max_length=config.ph,
        # saving_optimizer=config.ph,
        max_search_optim_loops=config.ph(float('inf')),
):
    filesys.asserts_conditional_exist(model_learning_dir_path, resuming)

    last_optim_dir_path = learning.get_last_optim_dir(model_learning_dir_path)
    best_optim_dir_path = learning.get_best_optim_dir(model_learning_dir_path)
    last_search_dir_path = learning.get_last_search_dir(model_learning_dir_path)
    best_search_dir_path = learning.get_best_search_dir(model_learning_dir_path)

    def get_init_optim_status():
        return get_init_status(measures=optim_measures, update_unit='optimization')

    def get_init_search_status():
        return get_init_status(measures=search_measures, update_unit='search')

    if resuming:
        optim_status = dict(learning.load_status(last_optim_dir_path, default=get_init_optim_status()))
        search_status = dict(learning.load_status(last_search_dir_path, default=get_init_search_status()))

        assert search_status['last_update_num'] == search_status['best_update_num']
        assert optim_status['last_update_num'] == optim_status['best_update_num']

        assert search_status['last_update_num'] - 1 <= optim_status['last_update_num'] <= search_status['last_update_num']
        optim_first = optim_status['last_update_num'] + 1 == search_status['last_update_num']
    else:
        optim_status = get_init_optim_status()
        search_status = get_init_search_status()
        optim_first = False

    def get_model_path():
        if optim_status['last_update_num'] > 0:
            return last_optim_dir_path
        else:
            return filesys.asserts_exist(pretrained_model_name_or_path)

    def run_search():
        search_result = search_to_collect(
            grammar=grammar, compiler=compiler, model_path=get_model_path(),
            devices=devices,
            context=context, encoded_dataset=encoded_weaksup_search_set,
            batch_size=search_batch_size, num_beams=num_search_beams,
            generation_max_length=generation_max_length,
            constrained_decoding=constrained_decoding, using_arg_candidate=using_arg_candidate)

        with replace_dir_wlmp(last_search_dir_path) as temp_last_dir_path:
            skip_if_not_wlmp(temp_last_dir_path)

            learning.save_status(search_status, temp_last_dir_path)
            learning.save_performance(search_result['performance'], temp_last_dir_path)
            learning.save_weaksup_dataset(search_result['weaksup_examples'], temp_last_dir_path)

        updating_best = update_status(search_status, performance=search_result['performance'])

        if updating_best:
            copy_dir_wlmp(last_search_dir_path, best_search_dir_path, replacing=True)

        return updating_best

    def run_optimization():
        raise NotImplementedError

    if optim_first:
        optim_updating_best = run_optimization()
    else:
        optim_updating_best = True

    while True:
        if optim_status['last_update_num'] < max_search_optim_loops:
            logger.info('Maximum number of loops for search and optimization')
            break

        # Search
        search_updating_best = run_search()

        if not search_updating_best:
            logger.info('Early stopping after search')
            break

        # Optimization
        optim_updating_best = run_optimization()

        if not optim_updating_best:
            logger.info('Early stopping after optimization')
            break

    raise NotImplementedError


if __name__ == '__main__':
    if config.run_mode == 'train-default':
        run_train()
        # from dhnamlib.pylib.cProfiling import run_context
        # run_context('run_train(model_learning_dir_path=config.model_learning_dir_path)', sort='cumtime')
    if config.run_mode == 'train-for-multiple-decoding-strategies':
        run_train_for_multiple_decoding_strategies()
    elif config.run_mode == 'test-on-val-set':
        run_test(encoded_test_set=config.encoded_val_set, evaluating=True)
        # from dhnamlib.pylib.cProfiling import run_context
        # run_context('run_test(encoded_test_set=config.encoded_val_set, evaluating=True)', sort='cumtime')
    elif config.run_mode == 'test-on-test-set':
        run_test(encoded_test_set=config.encoded_test_set, evaluating=False)
    elif config.run_mode == 'oracle-test-on-val-set':
        run_oracle_test(encoded_test_set=config.encoded_val_set, evaluating=True)
    else:
        raise Exception(f'Unknown execution type "{config.run_mode}"')

    # from dhnamlib.pylib.cProfiling import run_context
    # tuple(config.items(lazy=False))
    # run_context('run_train(model_learning_dir_path=config.model_learning_dir_path)', sort='cumtime')
