
from configuration import config

from .model import is_finetuned, load_pretrained_model


@config
def train(
        pretrained_model_name_or_path=config.ph,
        grammar=config.ph):

    model = load_pretrained_model(
        pretrained_model_name_or_path,
        tokenizer=(None if is_finetuned(pretrained_model_name_or_path) else grammar.lf_tokenizer))
    pass


if __name__ == '__main__':
    train()

