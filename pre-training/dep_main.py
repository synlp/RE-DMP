from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
import subprocess

import numpy as np
import torch

from modules.optimization import BertAdam
from modules.schedulers import LinearWarmUpScheduler

from tqdm import tqdm
from dep_helper import get_label_list
from dep_model import DependencyParser
import datetime


def train(args):

    if args.use_bert and args.use_zen:
        raise ValueError('We cannot use both BERT and ZEN')

    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_name = './logs/log-' + now_time
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=log_file_name,
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    logger.info(vars(args))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
                                             world_size=args.world_size)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps // args.world_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.model_name is None:
        raise Warning('model name is not specified, the model will NOT be saved!')
    output_model_dir = os.path.join('./models', args.model_name + '_' + now_time)

    label_list = get_label_list(args.label_path)
    logger.info('# of tag types in train: %d: ' % (len(label_list) - 3))
    label_map = {label: i for i, label in enumerate(label_list, 1)}

    if args.continue_train:
        dep_parser = DependencyParser.load_model(args.bert_model)
    else:
        hpara = DependencyParser.init_hyper_parameters(args)
        dep_parser = DependencyParser(label_map, hpara, args.bert_model, from_pretrained=(not args.vanilla))

    load_data = dep_parser.load_data
    convert_examples_to_features = dep_parser.convert_examples_to_features
    feature2input = dep_parser.feature2input
    decode = dep_parser.decode
    save_model = dep_parser.save_model

    total_params = sum(p.numel() for p in dep_parser.parameters() if p.requires_grad)
    logger.info('# of trainable parameters: %d' % total_params)

    num_train_optimization_steps = int(
        112217077 / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.fp16:
        dep_parser.half()
    dep_parser.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            # from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        dep_parser = DDP(dep_parser)
    elif n_gpu > 1:
        dep_parser = torch.nn.DataParallel(dep_parser)

    param_optimizer = list(dep_parser.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        print("using fp16")
        try:
            from apex.optimizers import FusedAdam
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)

        if args.loss_scale == 0:
            model, optimizer = amp.initialize(dep_parser, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(dep_parser, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale=args.loss_scale)
        scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                          total_steps=num_train_optimization_steps)

    else:
        # num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    if not args.local_rank == -1:
        args.save_every_steps = args.save_every_steps // args.world_size

    global_step = 0

    if args.continue_train:
        optimizer.load_state_dict(torch.load(os.path.join(args.bert_model, 'optimizer.bin')))

    logger.info("***** Running training *****")
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Batch size = %d", args.train_batch_size)

    for epoch in range(int(args.num_train_epochs)):
        files = os.listdir(args.train_data_path)
        files_tmp = []
        for f in files:
            if f.startswith('wiki'):
                files_tmp.append(f)
        files = files_tmp
        np.random.shuffle(files)
        files = files[args.rank::args.world_size]
        logger.info('Epoch %d start' % (epoch+1))
        for file_index, file in enumerate(files):
            train_examples = load_data(os.path.join(args.train_data_path, file))
            np.random.shuffle(train_examples)
            dep_parser.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            with tqdm(total=int(len(train_examples) / args.train_batch_size),
                      disable=(args.rank not in [-1, 0] and args.world_size > 1)) as pbar:
                for step, start_index in enumerate(range(0, len(train_examples), args.train_batch_size)):
                    dep_parser.train()
                    batch_examples = train_examples[start_index: min(start_index +
                                                                     args.train_batch_size, len(train_examples))]
                    if len(batch_examples) == 0:
                        continue
                    train_features = convert_examples_to_features(batch_examples)

                    if len(train_features) == 0:
                        continue

                    input_ids, input_mask, l_mask, eval_mask, arcs, rels, ngram_ids, ngram_positions, \
                    segment_ids, valid_ids, rel_second, rel_third = feature2input(device, train_features)

                    loss = dep_parser(input_ids, segment_ids, input_mask, valid_ids, l_mask,
                                      ngram_ids, ngram_positions,
                                      arcs, rels,
                                      rel_second, rel_third)

                    # if np.isnan(loss.to('cpu').detach().numpy()):
                    #     raise ValueError('loss is nan!')
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    pbar.update(1)

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                            scheduler.step()
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                        if global_step % args.save_every_steps == 0 \
                                and (args.local_rank == -1
                                     or torch.distributed.get_rank() == 0
                                     or args.world_size <= 1):
                            current_step = global_step * args.world_size

                            if not os.path.exists(output_model_dir):
                                os.makedirs(output_model_dir)

                            # model_to_save = dep_parser.module if hasattr(dep_parser, 'module') else dep_parser
                            step_model_dir = os.path.join(output_model_dir, 'step_%08d' % current_step)
                            if not os.path.exists(step_model_dir):
                                os.mkdir(step_model_dir)

                            save_model(step_model_dir, args.bert_model, optimizer)

    step_model_dir = os.path.join(output_model_dir, 'final')
    if not os.path.exists(step_model_dir):
        os.mkdir(step_model_dir)

    save_model(step_model_dir, args.bert_model, optimizer)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The training data path. Should contain the .tsv files for the task.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The data path containing the sentences to be segmented")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        help="The output path of segmented file")
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_xlnet",
                        action='store_true',
                        help="Whether to use XLNet.")
    parser.add_argument("--use_zen",
                        action='store_true',
                        help="Whether to use ZEN.")
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_model", default=None, type=str,
                        help="")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_size",
                        default=128,
                        type=int,
                        help="The maximum candidate word size used by attention. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=None,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--rank", type=int, default=None, help="local_rank for distributed training on gpus")
    parser.add_argument('--init_method', type=str, default=None)

    parser.add_argument('--patient', type=int, default=3, help="Patient for the early stop.")
    parser.add_argument('--save_every_steps', type=int, default=10000, help="")
    parser.add_argument('--model_name', type=str, default=None, help="")
    parser.add_argument('--label_path', type=str, default=None, help="")

    parser.add_argument('--vocab_path', type=str, default=None, help="")
    parser.add_argument('--analysis_path', type=str, default=None, help="")

    parser.add_argument("--use_second_order", action='store_true')
    parser.add_argument("--use_third_order", action='store_true')

    parser.add_argument("--use_biaffine", action='store_true')
    parser.add_argument("--continue_train", action='store_true')

    parser.add_argument("--vanilla", action='store_true')

    args = parser.parse_args()

    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        args.world_size = int(os.environ['SLURM_NTASKS'])
    if 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        args.local_rank = int(os.environ['SLURM_LOCALID'])

    args.local_rank = -1
    args.rank = 1
    args.world_size = 1

    if args.do_train:
        train(args)


if __name__ == "__main__":
    main()
