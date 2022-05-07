from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from modules.optimization import BertAdam
from modules.schedulers import LinearWarmUpScheduler

from tqdm import tqdm, trange
from re_helper import get_label_list
from re_eval import compute_metrics, compute_micro_f1, semeval_official_eval
from re_model import RelationExtractor
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

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.model_name is None:
        raise ValueError('model name is not specified, the model will NOT be saved!')
    output_model_dir = os.path.join('./models', args.model_name + '_' + now_time)

    label_list = get_label_list(args.train_data_path)
    logger.info('# of relation types in train: %d: ' % (len(label_list) - 1))
    label_map = {label: i for i, label in enumerate(label_list, 1)}

    hpara = RelationExtractor.init_hyper_parameters(args)
    relation_extractor = RelationExtractor(label_map, hpara, args.bert_model, from_pretrained=(not args.vanilla))

    train_examples = relation_extractor.load_data(args.train_data_path)
    dev_examples = relation_extractor.load_data(args.dev_data_path)

    convert_examples_to_features = relation_extractor.convert_examples_to_features
    feature2input = relation_extractor.feature2input
    save_model = relation_extractor.save_model

    total_params = sum(p.numel() for p in relation_extractor.parameters() if p.requires_grad)
    logger.info('# of trainable parameters: %d' % total_params)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.fp16:
        relation_extractor.half()
    relation_extractor.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        relation_extractor = DDP(relation_extractor)
    elif n_gpu > 1:
        relation_extractor = torch.nn.DataParallel(relation_extractor)

    param_optimizer = list(relation_extractor.named_parameters())
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
            model, optimizer = amp.initialize(relation_extractor, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(relation_extractor, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale=args.loss_scale)
        scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                          total_steps=num_train_optimization_steps)

    else:
        # num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    best_epoch = -1
    best_dev_p = -1
    best_dev_r = -1
    best_dev_f = -1

    history = {'epoch': [], 'p': [], 'r': [], 'f': []}

    num_of_no_improvement = 0
    patient = args.patient

    global_step = 0

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            relation_extractor.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
                relation_extractor.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue

                train_features = convert_examples_to_features(batch_examples)

                input_ids, input_mask, entity_mark, labels, ngram_ids, ngram_positions, \
                segment_ids = feature2input(device, train_features)

                loss = relation_extractor(input_ids, segment_ids, input_mask,
                                          entity_mark=entity_mark, labels=labels,
                                          input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

                if np.isnan(loss.to('cpu').detach().numpy()):
                    raise ValueError('loss is nan!')
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
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            relation_extractor.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info('\n')

                eval_examples = dev_examples
                relation_extractor.eval()

                pred_scores = None
                out_label_ids = None

                id2label = {i: label for i, label in enumerate(label_list, 1)}
                for start_index in range(0, len(eval_examples), args.eval_batch_size):
                    eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                         len(eval_examples))]

                    eval_features = convert_examples_to_features(eval_batch_examples)

                    input_ids, input_mask, entity_mark, labels, ngram_ids, ngram_positions, \
                    segment_ids = feature2input(device, eval_features)

                    with torch.no_grad():
                        logits = relation_extractor(input_ids, segment_ids, input_mask,
                                                    entity_mark=entity_mark, labels=None,
                                                    input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

                    if pred_scores is None:
                        pred_scores = logits.detach().cpu().numpy()
                        out_label_ids = labels.detach().cpu().numpy()
                    else:
                        pred_scores = np.append(pred_scores, logits.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

                id2label[0] = '<UNK>'

                all_pred_ids = np.argmax(pred_scores, axis=1)
                all_gold_ids = out_label_ids

                all_pred = [id2label[label_id] for label_id in all_pred_ids]
                all_gold = [id2label[label_id] for label_id in all_gold_ids]

                prediction = all_pred

                if not os.path.exists(output_model_dir):
                    os.makedirs(output_model_dir)

                if args.task_name == 'semeval':
                    result = semeval_official_eval(all_pred, all_gold, output_model_dir)
                else:
                    result = compute_metrics(all_pred_ids, all_gold_ids, len(label_map), label_map['Other'])
                    result["micro-f1"] = compute_micro_f1(all_pred_ids, all_gold_ids, label_map, ignore_label='Other',
                                                          output_dir=output_model_dir)
                    result["f1"] = result["micro-f1"]

                logging.info(result)

                p, r, f = result["precision"], result["recall"], result['f1']

                report = 'dev: Epoch: %d, precision:%.2f, recall:%.2f, f1:%.2f' \
                         % (epoch+1, p, r, f)
                logger.info(report)
                history['epoch'].append(epoch)
                history['p'].append(p)
                history['r'].append(r)
                history['f'].append(f)

                output_eval_file = os.path.join(output_model_dir, "dev_report.txt")
                with open(output_eval_file, "a") as writer:
                    writer.write(report)
                    writer.write('\n')

                logger.info('\n')
                if history['f'][-1] > best_dev_f:
                    best_epoch = epoch + 1
                    best_dev_p = history['p'][-1]
                    best_dev_r = history['r'][-1]
                    best_dev_f = history['f'][-1]

                    num_of_no_improvement = 0

                    if args.model_name:
                        with open(os.path.join(output_model_dir, 'dev_result.txt'), "w") as writer:
                            writer.write('pred\tgold\te1\te2\t\\text\n\n')
                            all_labels = prediction
                            for example, labels in zip(dev_examples, all_labels):
                                words = example.text_a
                                gold_labels = example.label
                                line = '\t'.join([labels, gold_labels, example.e1, example.e2, words])
                                writer.write(line)
                                writer.write('\n')

                        save_model(output_model_dir, args.bert_model)
                        arg_file = os.path.join(output_model_dir, 'args.txt')
                        with open(arg_file, 'w', encoding='utf8') as f:
                            f.write(str(vars(args)))
                            f.write('\n')
                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

        best_report = "Epoch: %d, dev_p: %f, dev_r: %f, dev_f: %f, " % (
            best_epoch, best_dev_p, best_dev_r, best_dev_f)

        logger.info("\n=======best f at dev========")
        logger.info(best_report)
        logger.info("\n=======best f at dev========")

        if args.model_name is not None:
            output_eval_file = os.path.join(output_model_dir, "final_report.txt")
            with open(output_eval_file, "w") as writer:
                writer.write(str(total_params))
                writer.write('\n')
                writer.write(best_report)

            with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
                json.dump(history, f)
                f.write('\n')


def test(args):

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    relation_extractor = RelationExtractor.load_model(args.eval_model, device)

    eval_examples = relation_extractor.load_data(args.test_data_path)

    convert_examples_to_features = relation_extractor.convert_examples_to_features
    feature2input = relation_extractor.feature2input
    label_map = {v: k for k, v in relation_extractor.labelmap.items()}

    if args.fp16:
        relation_extractor.half()
    relation_extractor.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        relation_extractor = DDP(relation_extractor)
    elif n_gpu > 1:
        relation_extractor = torch.nn.DataParallel(relation_extractor)

    relation_extractor.to(device)

    relation_extractor.eval()

    pred_scores = None
    out_label_ids = None

    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]

        eval_features = convert_examples_to_features(eval_batch_examples)

        input_ids, input_mask, entity_mark, labels, ngram_ids, ngram_positions, \
        segment_ids = feature2input(device, eval_features)

        with torch.no_grad():
            logits = relation_extractor(input_ids, segment_ids, input_mask,
                                        entity_mark=entity_mark, labels=None,
                                        input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

        if pred_scores is None:
            pred_scores = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            pred_scores = np.append(pred_scores, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    label_map[0] = '<UNK>'

    all_pred_ids = np.argmax(pred_scores, axis=1)
    all_gold_ids = out_label_ids

    all_pred = [label_map[label_id] for label_id in all_pred_ids]
    all_gold = [label_map[label_id] for label_id in all_gold_ids]

    output_model_dir = os.path.join(args.eval_model, '../')

    if args.task_name == 'semeval':
        result = semeval_official_eval(all_pred, all_gold, output_model_dir)
    else:
        result = compute_metrics(all_pred_ids, all_gold_ids, len(label_map), label_map['Other'])
        result["micro-f1"] = compute_micro_f1(all_pred_ids, all_gold_ids, label_map, ignore_label='Other',
                                              output_dir=output_model_dir)
        result["f1"] = result["micro-f1"]

    logging.info(result)

    p, r, f = result["precision"], result["recall"], result['f1']

    report = '%s P: %.2f R: %.2f f1: %.2f' % (args.test_data_path, p, r, f)
    print(report)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The training data path. Should contain the .tsv files for the task.")
    parser.add_argument("--dev_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--test_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--brown_data_path", default=None, type=str)
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
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="One of semeval, tacred, ace05, kbp37")
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
                        default=-1,
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
    parser.add_argument("--rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument('--init_method', type=str, default=None)

    parser.add_argument('--patient', type=int, default=3, help="Patient for the early stop.")
    parser.add_argument('--model_name', type=str, default=None, help="")
    parser.add_argument('--mlp_dropout', type=float, default=0.33, help='')
    parser.add_argument('--n_mlp', type=int, default=200, help='')

    parser.add_argument("--vanilla", action='store_true')

    args = parser.parse_args()

    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        args.world_size = int(os.environ['SLURM_NTASKS'])
    if 'RANK' in os.environ:
        assert args.local_rank == -1
        args.rank = int(os.environ['RANK'])
    elif 'SLURM_PROCID' in os.environ:
        assert args.local_rank == -1
        args.rank = int(os.environ['SLURM_PROCID'])
    if 'LOCAL_RANK' in os.environ:
        assert args.local_rank == -1
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        assert args.local_rank == -1
        args.local_rank = int(os.environ['SLURM_LOCALID'])

    if args.init_method is None:
        if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
            args.init_method = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
        else:
            args.init_method = "tcp://127.0.0.1:23456"

    args.local_rank = -1

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    # elif args.do_predict:
    #     predict(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval`, `do_predict` must be True.')


if __name__ == "__main__":
    main()
