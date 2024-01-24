
import os
# import warnings
import glob

import torch

from configuration import config

from . import learning
from .data_read import make_data_loader
from .validation import validate
from .learning import optim_measures, search_measures


from dhnamlib.pylib import filesys
from dhnamlib.pylib.iteration import not_none_valued_dict
from dhnamlib.pylib.mllib.learning import get_init_status, update_status
from dhnamlib.pylib.function import identity
from dhnamlib.pylib.hflib.acceleration import broadcast_object


MODEL_SYMLINK_NAME = 'model'
# COLLECTION_SYMLINK_NAME = 'collection'
# LEARNING_SYMLINK_NAME = 'learning'


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
        train_max_num_action_seqs=config.ph(None),
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
        weaksup_learning=config.ph(False),
):
    if restarting:
        assert learning.is_finetuned(pretrained_model_name_or_path)

        if model_learning_dir_path is None:
            model_learning_dir_path = filesys.get_parent_path(pretrained_model_name_or_path)
    else:
        assert model_learning_dir_path is not None

    learning.save_config_info(model_learning_dir_path)

    last_dir_path = os.path.join(model_learning_dir_path, 'last')
    # mkloc_unless_exist_wlmp(last_dir_path)
    # make_symlink_wlmp(learning.get_new_checkpoint_path(model_learning_dir_path), last_dir_path)

    best_dir_path = os.path.join(model_learning_dir_path, 'best')
    # mkloc_unless_exist_wlmp(best_dir_path)
    # copy_symlink_wlmp(last_dir_path, best_dir_path)

    cpm = learning.AcceleratedCheckpointManager(
        checkpoint_loc_path=os.path.join(model_learning_dir_path, 'checkpoint'),
        symlink_glob_patterns=[os.path.join(model_learning_dir_path, 'last', MODEL_SYMLINK_NAME),
                               os.path.join(model_learning_dir_path, 'best', MODEL_SYMLINK_NAME)])

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
    model, train_data_loader, optimizer, scheduler = config.accelerator.prepare(
        model, train_data_loader, optimizer, scheduler)

    val_data_loader = config.accelerator.prepare_val_data_loader(val_data_loader)

    remaining_patience = patience

    forward_backward = learning.ws_forward_backward if weaksup_learning else learning.ss_forward_backward

    for epoch in range(status['last_update_num'] + 1, num_train_epochs + 1):
        logger.info(f'Epoch {epoch} starts')
        model.train()

        # debug_batch_idx = -1
        # loss = torch.tensor(0.)
        loss = 0
        # for batch in config.xtqdm(train_data_loader, desc_fn=lambda: 'loss: {:7.4f}'.format(loss.item())):
        for batch in config.utqdm(train_data_loader, unit='loss', update_fn=lambda: loss, repr_format='{:7.4f}'):
            # if debug_batch_idx > 100:
            #     break
            # else:
            #     debug_batch_idx += 1

            # - `model.config.decoder_start_token_id` is the first id of output sequences.
            # - the order or decoder output tokens in a sequence: decoder_start_token_id, bos_token_id, others ...

            optimizer.zero_grad()

            loss = forward_backward(**not_none_valued_dict(
                grammar=grammar,
                model=model,
                batch=batch,
                softmax_masking=None if weaksup_learning else softmax_masking,
                max_num_action_seqs=train_max_num_action_seqs if weaksup_learning else None,
            ))

            assert config.accelerator.sync_gradients
            if config.accelerator.sync_gradients:
                config.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

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

        new_checkpoint_dir_path = cpm.get_new_checkpoint_path()
        with learning.prepare_dir(new_checkpoint_dir_path) as temp_checkpoint_dir_path:
            learning.skip_if_not_wlmp(temp_checkpoint_dir_path)

            if saving_optimizer:
                learning.save_optimizer(optimizer, temp_checkpoint_dir_path)
            learning.save_scheduler(scheduler, temp_checkpoint_dir_path)
            learning.save_model(model, temp_checkpoint_dir_path)

        with cpm.cleaning(learning.replace_dir(last_dir_path)) as temp_last_dir_path:
            learning.skip_if_not_wlmp(temp_last_dir_path)

            learning.save_status(status, temp_last_dir_path)
            learning.save_performance(performance, temp_last_dir_path)
            learning.save_analysis(validation['analysis'], temp_last_dir_path)

            learning.make_symlink(
                new_checkpoint_dir_path,
                os.path.join(temp_last_dir_path, MODEL_SYMLINK_NAME))

        if updating_best:
            with cpm.cleaning(learning.replace_dir(best_dir_path)) as temp_best_dir_path:
                learning.skip_if_not_wlmp(temp_best_dir_path)

                learning.copy_dir(last_dir_path, temp_best_dir_path)

            remaining_patience = patience

            logger.info('Best model is updated')
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

    learning.save_config_info(model_learning_dir_path)

    common_last_dir_path = os.path.join(model_learning_dir_path, 'common:last')

    def get_best_dir_path(decoding_strategy_name):
        return os.path.join(model_learning_dir_path, f'{decoding_strategy_name}:best')

    def get_last_dir_path(decoding_strategy_name):
        return os.path.join(model_learning_dir_path, f'{decoding_strategy_name}:last')

    cpm = learning.AcceleratedCheckpointManager(
        checkpoint_loc_path=os.path.join(model_learning_dir_path, 'checkpoint'),
        symlink_glob_patterns=[os.path.join(model_learning_dir_path, '*:last', MODEL_SYMLINK_NAME),
                               os.path.join(model_learning_dir_path, '*:best', MODEL_SYMLINK_NAME)])

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
    model, train_data_loader, optimizer, scheduler = config.accelerator.prepare(
        model, train_data_loader, optimizer, scheduler)

    val_data_loader = config.accelerator.prepare_val_data_loader(val_data_loader)

    assert not softmax_masking, 'The feature of softmax_masking is not implemented'

    for epoch in range(1, num_train_epochs + 1):
        logger.info(f'Epoch {epoch} starts')
        model.train()

        # debug_batch_cnt = -1

        # loss = torch.tensor(0.)
        loss = 0
        # for batch in config.xtqdm(train_data_loader, desc_fn=lambda: 'loss: {:7.4f}'.format(loss.item())):
        for batch in config.utqdm(train_data_loader, unit='loss', update_fn=lambda: loss, repr_format='{:7.4f}'):
            # debug_batch_cnt += 1
            # if debug_batch_cnt > 100:
            #     break

            optimizer.zero_grad()

            loss = learning.ss_forward_backward(
                grammar=grammar,
                model=model,
                batch=batch,
                softmax_masking=softmax_masking,
            )

            assert config.accelerator.sync_gradients
            if config.accelerator.sync_gradients:
                config.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

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
                    last_model_saved = True

                    # save a model
                    new_checkpoint_dir_path = cpm.get_new_checkpoint_path()
                    with learning.prepare_dir(new_checkpoint_dir_path) as temp_checkpoint_dir_path:
                        learning.skip_if_not_wlmp(temp_checkpoint_dir_path)

                        if saving_optimizer:
                            learning.save_optimizer(optimizer, temp_checkpoint_dir_path)
                        learning.save_scheduler(scheduler, temp_checkpoint_dir_path)
                        learning.save_model(model, temp_checkpoint_dir_path)

                    with cpm.cleaning(learning.replace_dir(common_last_dir_path)) as temp_common_last_dir_path:
                        learning.skip_if_not_wlmp(temp_common_last_dir_path)

                        learning.make_symlink(
                            new_checkpoint_dir_path,
                            os.path.join(temp_common_last_dir_path, MODEL_SYMLINK_NAME))

                # save for a decoding strategy
                strategy_last_dir_path = os.path.join(model_learning_dir_path, f'{config.decoding_strategy_name}:last')
                with cpm.cleaning(learning.replace_dir(strategy_last_dir_path)) as temp_strategy_last_dir_path:
                    learning.skip_if_not_wlmp(temp_strategy_last_dir_path)

                    learning.save_status(status, temp_strategy_last_dir_path)
                    learning.save_performance(performance, temp_strategy_last_dir_path)
                    learning.save_analysis(validation['analysis'], temp_strategy_last_dir_path)

                    learning.copy_symlink(os.path.join(common_last_dir_path, MODEL_SYMLINK_NAME),
                                          os.path.join(temp_strategy_last_dir_path, MODEL_SYMLINK_NAME))

                if updating_best:
                    strategy_best_dir_path = os.path.join(model_learning_dir_path, f'{config.decoding_strategy_name}:best')
                    with cpm.cleaning(learning.replace_dir(strategy_best_dir_path)) as temp_strategy_best_dir_path:
                        learning.skip_if_not_wlmp(temp_strategy_best_dir_path)

                        learning.copy_dir(strategy_last_dir_path, temp_strategy_best_dir_path)

                    logger.info('Best model is updated')

                logger.info(f'Results are saved in "{model_learning_dir_path}"')


@config
def run_test(
        *,
        model_learning_dir_path=config.ph(None),
        model_path=config.ph(None),
        model_dir_name=config.ph('best'),
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
    if model_path is None:
        # Check the default checkpoint path
        assert model_learning_dir_path is not None
        assert os.path.isdir(model_learning_dir_path), f'The learning directory {model_learning_dir_path} does not exist'
        model_path = os.path.join(model_learning_dir_path, model_dir_name, MODEL_SYMLINK_NAME)
        assert os.path.isdir(model_path), f'The checkpoint {model_path} does not exist'
    model = learning.load_model(
        model_path,
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

    learning.mkloc_unless_exist(test_dir_path)
    learning.save_config_info(test_dir_path)

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
        # model_path=config.ph(None),
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
        # model_path=model_path,
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


@config
def run_search_train(
        pretrained_model_path=config.ph,
        grammar=config.ph,
        compiler=config.ph,
        # # devices=config.ph,
        logger=config.ph,
        # encoded_train_set=config.ph,
        encoded_val_set=config.ph,
        encoded_weaksup_search_set=config.ph,  # New
        train_batch_size=config.ph,
        val_batch_size=config.ph,
        search_batch_size=config.ph,  # New
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
        resuming=config.ph(None),         # New
        # # max_num_iterations=None,  # New
        context=config.ph,
        num_prediction_beams=config.ph,
        num_search_beams=config.ph,  # New
        generation_max_length=config.ph,
        # # saving_optimizer=config.ph,
        max_search_optim_loops=config.ph(float('inf')),  # New
):
    if resuming:
        logger.info(f'Learning resumes with the directory "{model_learning_dir_path}"')

    search_dir_path = os.path.join(model_learning_dir_path, 'search')
    optim_dir_path = os.path.join(model_learning_dir_path, 'optim')

    filesys.asserts_conditional_exist(
        os.path.join(search_dir_path, 'last'), resuming,
        f'When resuming == True, {model_learning_dir_path} should contain previous result of learning',
        f'When resuming == False, {model_learning_dir_path} should not contain any previous result of learning',
    )

    search_cpm = learning.AcceleratedCheckpointManager(
        checkpoint_loc_path=os.path.join(search_dir_path, 'checkpoint'),
        symlink_glob_patterns=[os.path.join(search_dir_path, 'last'),
                               os.path.join(search_dir_path, 'best')])
    optim_cpm = learning.AcceleratedCheckpointManager(
        checkpoint_loc_path=os.path.join(optim_dir_path, 'checkpoint'),
        symlink_glob_patterns=[os.path.join(optim_dir_path, 'last'),
                               os.path.join(optim_dir_path, 'best')])

    def get_init_search_status():
        return get_init_status(measures=search_measures, update_unit='search')

    def get_init_optim_status():
        return get_init_status(measures=optim_measures, update_unit='optimization')

    if resuming:
        search_status = learning.load_status(os.path.join(search_dir_path, 'last'), default=get_init_search_status())
        optim_status = learning.load_status(os.path.join(optim_dir_path, 'last'), default=get_init_optim_status())

        assert search_status['last_update_num'] == search_status['best_update_num']
        assert optim_status['last_update_num'] == optim_status['best_update_num']

        assert search_status['last_update_num'] - 1 <= optim_status['last_update_num'] <= search_status['last_update_num']
        optim_first = optim_status['last_update_num'] + 1 == search_status['last_update_num']
    else:
        search_status = get_init_search_status()
        optim_status = get_init_optim_status()
        optim_first = False

    weaksup_search_data_loader = config.accelerator.prepare_val_data_loader(make_data_loader(
        encoded_dataset=encoded_weaksup_search_set,
        decoder_start_token_id=grammar.model_config.decoder_start_token_id,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=search_batch_size,
        shuffle=False,
    ))

    def get_latest_model_path():
        if optim_status['last_update_num'] > 0:
            return os.path.join(optim_dir_path, 'last', 'best', MODEL_SYMLINK_NAME)
        else:
            return filesys.asserts_exist(pretrained_model_path)

    def load_latest_model():
        model = learning.load_model(get_latest_model_path(), num_tokens=len(grammar.lf_tokenizer))
        model.to(config.accelerator.device)
        model = config.accelerator.prepare(model)
        return model

    def load_latest_encoded_weaksup_set():
        return learning.load_weaksup_dataset(os.path.join(search_dir_path, 'last'))

    def run_search():
        config.accelerator.wait_for_everyone()  # before loading model
        model = load_latest_model()
        model.eval()

        logger.info('Search starts')

        validation = validate(
            grammar=grammar,
            compiler=compiler,
            model=model,
            context=context,
            data_loader=weaksup_search_data_loader,
            batch_size=search_batch_size,
            num_beams=num_search_beams,
            generation_max_length=generation_max_length,
            analyzing=False,
            softmax_masking=softmax_masking,
            constrained_decoding=constrained_decoding,
            using_arg_candidate=using_arg_candidate,
            evaluating=True,
            using_oracle=True,
            collecting_weaksup_examples=True,
            strict_postprocessing=True,
        )

        performance = validation['performance']
        updating_best = update_status(search_status, performance=performance)

        logger.info(f'Search update number: {search_status["last_update_num"]} / Performance: {str(performance)}')

        new_checkpoint_dir_path = search_cpm.get_new_checkpoint_path()
        with learning.prepare_dir(new_checkpoint_dir_path) as temp_checkpoint_dir_path:
            learning.skip_if_not_wlmp(temp_checkpoint_dir_path)

            learning.save_status(search_status, temp_checkpoint_dir_path)
            learning.save_performance(performance, temp_checkpoint_dir_path)
            learning.save_weaksup_dataset(validation['weaksup_examples'], temp_checkpoint_dir_path)
            learning.save_time_info(validation['time_info'], temp_checkpoint_dir_path)
            learning.save_predictions(validation['predictions'], temp_checkpoint_dir_path)

        last_dir_path = os.path.join(search_dir_path, 'last')
        learning.change_symlink(new_checkpoint_dir_path, last_dir_path)
        search_cpm.clean()

        if updating_best:
            best_dir_path = os.path.join(search_dir_path, 'best')
            learning.copy_symlink(last_dir_path, best_dir_path)
            search_cpm.clean()

            logger.info('Best search result is updated')

        return updating_best

    def run_optim():
        config.accelerator.wait_for_everyone()  # before loading encoded_weaksup_set
        encoded_weaksup_set = load_latest_encoded_weaksup_set()

        new_checkpoint_dir_path = broadcast_object(optim_cpm.get_new_checkpoint_path())

        # checkpoint_loc_path = os.path.join(optim_dir_path, 'checkpoint')
        # learning.mkloc_unless_exist(checkpoint_loc_path)
        # temp_checkpoint_dir_path = broadcast_object(learning.mkdtemp(dir=checkpoint_loc_path))

        logger.info('Optimization starts')

        run_train(
            pretrained_model_name_or_path=get_latest_model_path(),
            grammar=grammar,
            compiler=compiler,
            logger=logger,
            encoded_train_set=encoded_weaksup_set,
            encoded_val_set=encoded_val_set,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            using_scheduler=using_scheduler,
            num_warmup_epochs=num_warmup_epochs,
            max_grad_norm=max_grad_norm,
            patience=patience,
            softmax_masking=softmax_masking,
            constrained_decoding=constrained_decoding,
            using_arg_candidate=using_arg_candidate,
            model_learning_dir_path=new_checkpoint_dir_path,
            restarting=False,
            context=context,
            num_prediction_beams=num_prediction_beams,
            generation_max_length=generation_max_length,
            saving_optimizer=False,
            weaksup_learning=True,
        )

        # new_checkpoint_dir_path = optim_cpm.get_new_checkpoint_path()
        # learning.rename_dir(temp_checkpoint_dir_path, new_checkpoint_dir_path)

        config.accelerator.wait_for_everyone()  # before loading performance

        performance = learning.load_performance(os.path.join(new_checkpoint_dir_path, 'best'))
        updating_best = update_status(optim_status, performance=performance)

        learning.save_status(optim_status, new_checkpoint_dir_path)  # save the status after update

        last_dir_path = os.path.join(optim_dir_path, 'last')
        learning.change_symlink(new_checkpoint_dir_path, last_dir_path)
        optim_cpm.clean()

        logger.info(f'Optimization update number: {optim_status["last_update_num"]} / Performance: {str(performance)}')

        if updating_best:
            best_dir_path = os.path.join(optim_dir_path, 'best')
            learning.copy_symlink(last_dir_path, best_dir_path)
            optim_cpm.clean()

            logger.info('Best optimization result is updated')

        return updating_best

    if optim_first:
        optim_updating_best = run_optim()
    else:
        optim_updating_best = True

    while optim_status['last_update_num'] < max_search_optim_loops:
        # Search
        search_updating_best = run_search()

        if not search_updating_best:
            logger.info('Early stopping after search')
            break

        # Optimization
        optim_updating_best = run_optim()

        # if not optim_updating_best:
        #     logger.info('Early stopping after optimization')
        #     break
    else:
        logger.info('Maximum number of loops for search and optimization')


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
    elif config.run_mode == 'search-train':
        run_search_train()
    elif config.run_mode == 'test-on-test-set':
        run_test(encoded_test_set=config.encoded_test_set, evaluating=False)
    elif config.run_mode == 'oracle-test-on-val-set':
        run_oracle_test(encoded_test_set=config.encoded_val_set, evaluating=True)
    else:
        raise Exception(f'Unknown execution type "{config.run_mode}"')

    # from dhnamlib.pylib.cProfiling import run_context
    # tuple(config.items(lazy=False))
    # run_context('run_train(model_learning_dir_path=config.model_learning_dir_path)', sort='cumtime')
