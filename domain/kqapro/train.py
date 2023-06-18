
import os
import torch

from configuration import config

from .model import is_finetuned, load_model
from .data_read import make_data_loader

from dhnamlib.pylib import filesys
from dhnamlib.pylib.torchlib.optimization import get_linear_schedule_with_warmup


OPTIMIZER_FILE_NAME = 'optimizer.pt'
SCHEDULER_FILE_NAME = 'scheduler.pt'


@config
def train(
        pretrained_model_name_or_path=config.ph,
        grammar=config.ph,
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
        output_dir=None,
        restarting=False,
):
    if restarting:
        assert is_finetuned(pretrained_model_name_or_path)

        prev_optimizer_file_path = os.path.join(pretrained_model_name_or_path, OPTIMIZER_FILE_NAME)
        assert os.path.isfile(prev_optimizer_file_path)

        prev_scheduler_file_path = os.path.join(pretrained_model_name_or_path, SCHEDULER_FILE_NAME)
        assert os.path.isfile(prev_scheduler_file_path)

        if output_dir is None:
            output_dir = filesys.get_parent_path(pretrained_model_name_or_path)

    model = load_model(
        pretrained_model_name_or_path,
        num_tokens=len(grammar.lf_tokenizer))
    train_data_loader = make_data_loader(
        encoded_dataset=encoded_train_set,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=batch_size,
        shuffle=True)
    val_data_loader = make_data_loader(
        encoded_dataset=encoded_val_set,
        pad_token_id=grammar.lf_tokenizer.pad_token_id,
        batch_size=batch_size * 4,
        shuffle=False)

    non_decayed_names = ["bias", "layernorm", "layer_norm"]
    model_named_parameters = tuple(model.named_parameters())

    def is_decayed_param_name(param_name):
        return not any(nd_name in param_name for nd_name in non_decayed_names)

    # 'weight_decay' is for AdamW
    # 'lr' values become LambdaLR's base_lrs which are multiplied by lr_lambdas
    param_groups = [
        dict(params=[param for param_name, param in model_named_parameters
                     if is_decayed_param_name(param_name)],
             weight_decay=weight_decay,
             lr=learning_rate),
        dict(params=[param for param_name, param in model_named_parameters
                     if not is_decayed_param_name(param_name)],
             weight_decay=0.0,
             lr=learning_rate)]

    num_training_steps = len(train_data_loader) * num_train_epochs

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    if restarting:
        optimizer.load_state_dict(torch.load(prev_optimizer_file_path))
        scheduler.load_state_dict(torch.load(prev_scheduler_file_path))

    model.to(device)
    
    pass


if __name__ == '__main__':
    train()

