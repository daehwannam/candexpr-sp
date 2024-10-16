import os
import time
# import torch.optim as optim
# import torch.nn as nn
import argparse
import json
import re
import warnings

import torch
from tqdm import tqdm
# from datetime import date
# from kqapro_util.misc import MetricLogger, seed_everything, ProgressBar
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
# import torch.optim as optim
# from kqapro_util.lr_scheduler import get_linear_schedule_with_warmup
from kopl.kopl import KoPLEngine
from kqapro_util.misc import seed_everything

from dhnamlib.pylib.filesys import make_logger
from dhnamlib.pylib.time import TimeMeasure

from .data import DataLoader
from .util import register

# warnings.simplefilter("ignore")  # hide warnings that caused by invalid sparql query
# from termcolor import colored


def post_process(text):
    pattern = re.compile(r'".*?"')
    nes = []
    for item in pattern.finditer(text):
        nes.append((item.group(), item.span()))
    pos = [0]
    for name, span in nes:
        pos += [span[0], span[1]]
    pos.append(len(text))
    assert len(pos) % 2 == 0
    assert len(pos) / 2 == len(nes) + 1
    chunks = [text[pos[i]: pos[i+1]] for i in range(0, len(pos), 2)]
    for i in range(len(chunks)):
        chunks[i] = chunks[i].replace('?', ' ?').replace('.', ' .')
    bingo = ''
    for i in range(len(chunks) - 1):
        bingo += chunks[i] + nes[i][0]
    bingo += chunks[-1]
    return bingo


def vis(args, kb, model, data, device, tokenizer):
    while True:
        text = input('Input your question:')
        with torch.no_grad():
            input_ids = tokenizer.batch_encode_plus([text], max_length = 512, pad_to_max_length = True, return_tensors="pt", truncation = True)
            source_ids = input_ids['input_ids'].to(device)
            outputs = model.generate(
                input_ids=source_ids,
                max_length = 500,
            )
            outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs]
            outputs = [post_process(output) for output in outputs]
            print(outputs[0])


def predict(args, model, data, device, tokenizer, executor, postprocessing_answer: bool, num_beams=1, verbose=False):
    model.eval()
    count, correct = 0, 0
    with torch.no_grad():
        all_outputs = []
        total_decoding_time = 0
        tm = TimeMeasure()
        for batch in tqdm(data, total=len(data)):
            source_ids = batch[0].to(device)
            tm.check()
            outputs = model.generate(
                input_ids=source_ids,
                max_length = 500,
                num_beams=num_beams,
            )
            total_decoding_time += tm.elapse()

            all_outputs.extend(outputs.cpu().numpy())

        logger = register.retrieve('logger')
        if verbose:
            logger.info('Total decoding time: {} second'.format(total_decoding_time))
            logger.info('Average decoding time: {} second'.format(total_decoding_time / len(data.dataset)))
            logger.info('Average decoding time: {:.2f} millisecond'.format((total_decoding_time * 1000) / len(data.dataset)))

        outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
        with open(os.path.join(args.save_dir, 'predict.txt'), 'w') as f:

            for output in tqdm(outputs):
                chunks = output.split('<func>')
                func_list = []
                inputs_list = []
                for chunk in chunks:
                    chunk = chunk.strip()
                    res = chunk.split('<arg>')
                    res = [_.strip() for _ in res]
                    if len(res) > 0:
                        func = res[0]
                        inputs = []
                        if len(res) > 1:
                            for x in res[1:]:
                                inputs.append(x)
                        else:
                            inputs = []
                        func_list.append(func)
                        inputs_list.append(inputs)
                ans = executor.forward(func_list, inputs_list, ignore_error = True)
                if ans is None:
                    ans = 'no'
                elif isinstance(ans, list) and len(ans) > 0:
                    ans = sorted(ans)[0]
                elif isinstance(ans, list) and len(ans) == 0:
                    if postprocessing_answer:
                        ans = 'no'
                    else:
                        ans = 'None'
                f.write(ans + '\n')
                
def validate(model, data, device, tokenizer, executor, postprocessing_answer: bool, num_beams=1, verbose=False):
    model.eval()
    count, correct = 0, 0
    with torch.no_grad():
        all_outputs = []
        all_answers = []
        total_decoding_time = 0
        tm = TimeMeasure()
        for batch in tqdm(data, total=len(data)):
            source_ids, source_mask, choices, target_ids, answer = [x.to(device) for x in batch]
            tm.check()
            batch_outputs = model.generate(
                input_ids=source_ids,
                max_length = 500,
                num_beams=num_beams,
            )
            total_decoding_time += tm.elapse()

            all_outputs.extend(batch_outputs.cpu().numpy())
            all_answers.extend(answer.cpu().numpy())

        logger = register.retrieve('logger')
        if verbose:
            logger.info('Total decoding time: {} second'.format(total_decoding_time))
            logger.info('Average decoding time: {} second'.format(total_decoding_time / len(data.dataset)))
            logger.info('Average decoding time: {:.2f} millisecond'.format((total_decoding_time * 1000) / len(data.dataset)))
        
        outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
        given_answer = [data.vocab['answer_idx_to_token'][a] for a in all_answers]
        for a, output in tqdm(zip(given_answer, outputs)):
            chunks = output.split('<func>')
            func_list = []
            inputs_list = []
            for chunk in chunks:
                chunk = chunk.strip()
                res = chunk.split('<arg>')
                res = [_.strip() for _ in res]
                if len(res) > 0:
                    func = res[0]
                    inputs = []
                    if len(res) > 1:
                        for x in res[1:]:
                            inputs.append(x)
                    else:
                        inputs = []
                    func_list.append(func)
                    inputs_list.append(inputs)
            ans = executor.forward(func_list, inputs_list, ignore_error = True)
            if ans is None:
                ans = 'no'
            elif isinstance(ans, list) and len(ans) > 0:
                ans = sorted(ans)[0]
            elif isinstance(ans, list) and len(ans) == 0:
                if postprocessing_answer:
                    ans = 'no'
            if ans == a:
                correct += 1
            count += 1
        acc = correct / count
        logger = register.retrieve('logger')
        logger.info('acc: {}'.format(acc))

        return acc


def train(args):
    logger = register.retrieve('logger')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    val_pt = os.path.join(args.input_dir, 'test.pt')
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    logger.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.ckpt)
    model = model_class.from_pretrained(args.ckpt)
    model = model.to(device)
    logger.info(model)
    engine = KoPLEngine(json.load(open(os.path.join(args.input_dir, 'kb.json'))))
    # validate(model, val_loader, device, tokenizer, engine)

    predict(args, model, val_loader, device, tokenizer, engine, args.postprocessing_answer, args.num_beams, args.verbose)


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', required=True)

    # training parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--postprocessing-answer', dest='postprocessing_answer', action='store_true', help='post-processing answers')
    parser.add_argument('--num_beams', dest='num_beams', default=1, type=int, help='beam size')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='post-processing answers')
    
    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)
    # parser.add_argument('--top_p', default=)
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1e-4, type = float)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    register('logger', make_logger('train', os.path.join(args.save_dir, '{}.log'.format(current_time))))
    logger = register.retrieve('logger')

    # args display
    for k, v in vars(args).items():
        logger.info(k+':'+str(v))

    seed_everything(42)

    train(args)


if __name__ == '__main__':
    main()
