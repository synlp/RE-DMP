from __future__ import absolute_import, division, print_function

import copy
import math

import numpy as np
import torch
from torch import nn
from modules import BertModel, BertTokenizer, Biaffine, MLP
from transformers_xlnet import XLNetModel, XLNetTokenizer
from util import eisner, ispunct
import re
import subprocess
import os
from dep_helper import load_json, save_json


DEFAULT_HPARA = {
    'max_seq_length': 128,
    'use_bert': False,
    'use_xlnet': False,
    'do_lower_case': False,
}


class DependencyParser(nn.Module):

    def __init__(self, labelmap, hpara, model_path, from_pretrained=True):
        super().__init__()
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']
        self.arc_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.rel_criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.tokenizer = None
        self.bert = None
        self.transformer = None

        if self.hpara['use_bert']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            if from_pretrained:
                self.bert = BertModel.from_pretrained(model_path, cache_dir='')
            else:
                from modules import CONFIG_NAME, BertConfig
                config_file = os.path.join(model_path, CONFIG_NAME)
                config = BertConfig.from_json_file(config_file)
                self.bert = BertModel(config)
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_xlnet']:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            if from_pretrained:
                self.transformer = XLNetModel.from_pretrained(model_path)
                state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
                key_list = list(state_dict.keys())
                reload = False
                for key in key_list:
                    if key.find('xlnet.') > -1:
                        reload = True
                        state_dict[key[key.find('xlnet.') + len('xlnet.'):]] = state_dict[key]
                    state_dict.pop(key)
                if reload:
                    self.xlnet.load_state_dict(state_dict)
            else:
                config, model_kwargs = XLNetModel.config_class.from_pretrained(model_path, return_unused_kwargs=True)
                self.transformer = XLNetModel(config)
            hidden_size = self.transformer.config.hidden_size
            self.dropout = nn.Dropout(self.transformer.config.summary_last_dropout)
        else:
            raise ValueError()

        self.linear_arc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rel_classifier_1 = nn.Linear(hidden_size, self.num_labels, bias=False)
        self.rel_classifier_2 = nn.Linear(hidden_size, self.num_labels, bias=False)
        self.bias = nn.Parameter(torch.tensor(self.num_labels, dtype=torch.float), requires_grad=True)
        nn.init.zeros_(self.bias)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None,
                attention_mask_label=None,
                input_ngram_ids=None, ngram_position_matrix=None,
                arcs=None, rels=None,
                ):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.transformer is not None:
            transformer_outputs = self.transformer(input_ids, token_type_ids, attention_mask=attention_mask)
            sequence_output = transformer_outputs[0]
        else:
            raise ValueError()

        batch_size, _, feat_dim = sequence_output.shape
        max_len = attention_mask_label.shape[1]
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=input_ids.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            sent_len = attention_mask_label[i].sum()
            # valid_output[i][:temp.size(0)] = temp
            valid_output[i][:sent_len] = temp[:sent_len]

        valid_output = self.dropout(valid_output)

        # get arc and rel scores from the bilinear attention
        # tmp_arc = self.linear_arc(valid_output).permute(0, 2, 1)
        # [batch_size, seq_len, seq_len]
        tmp_arc = self.linear_arc(valid_output).permute(0, 2, 1)
        s_arc = torch.bmm(valid_output, tmp_arc)

        # [batch_size, seq_len, seq_len, n_rels]
        rel_1 = self.rel_classifier_1(valid_output)
        rel_2 = self.rel_classifier_2(valid_output)
        rel_1 = torch.stack([rel_1] * max_len, dim=1)
        rel_2 = torch.stack([rel_2] * max_len, dim=2)
        s_rel = rel_1 + rel_2 + self.bias
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~attention_mask_label.unsqueeze(1), float('-inf'))

        if arcs is None:
            return s_arc, s_rel
        else:
            attention_mask_label[:, 0] = 0
            return self.get_loss(s_arc, s_rel, arcs, rels, attention_mask_label)

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_xlnet'] = args.use_xlnet
        hyper_parameters['do_lower_case'] = args.do_lower_case

        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def load_model(cls, model_path):
        tag_file = os.path.join(model_path, 'labelset.json')
        labelmap = load_json(tag_file)

        hpara_file = os.path.join(model_path, 'hpara.json')
        hpara = load_json(hpara_file)
        DEFAULT_HPARA.update(hpara)

        res = cls(labelmap=labelmap, hpara=DEFAULT_HPARA, model_path=model_path)
        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))

        return res

    def save_model(self, output_dir, vocab_dir, optimizer=None):

        output_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        output_tag_file = os.path.join(output_dir, 'labelset.json')
        save_json(output_tag_file, self.labelmap)

        output_hpara_file = os.path.join(output_dir, 'hpara.json')
        save_json(output_hpara_file, self.hpara)

        output_config_file = os.path.join(output_dir, 'config.json')
        with open(output_config_file, "w", encoding='utf-8') as writer:
            if self.bert:
                writer.write(self.bert.config.to_json_string())
            elif self.transformer:
                writer.write(self.transformer.config.to_json_string())
        output_bert_config_file = os.path.join(output_dir, 'bert_config.json')
        command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
        subprocess.run(command, shell=True)

        if self.bert:
            vocab_name = 'vocab.txt'
        elif self.transformer:
            vocab_name = 'spiece.model'
        else:
            raise ValueError()
        vocab_path = os.path.join(vocab_dir, vocab_name)
        command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(output_dir, vocab_name))
        subprocess.run(command, shell=True)

        if optimizer is not None:
            output_model_path = os.path.join(output_dir, 'optimizer.bin')
            torch.save(optimizer.state_dict(), output_model_path)

    def load_data(self, data_path):
        lines = readfile(data_path)

        examples = self.process_data(lines)

        return examples

    @staticmethod
    def process_data(lines):

        examples = []
        for i, (sentence, head, label) in enumerate(lines):
            guid = "%s" % str(i)
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None, head=head,
                                         label=label))
        return examples

    def get_loss(self, arc_scores, rel_scores, arcs, rels, mask, order='first'):
        arc_scores, arcs = arc_scores[mask], arcs[mask]
        rel_scores, rels = rel_scores[mask], rels[mask]

        if order == 'first':
            rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
            arc_loss = self.arc_criterion(arc_scores, arcs)
        else:
            rel_scores = rel_scores[arcs == 1]
            rels = rels[arcs == 1]

            arc_scores = torch.flatten(arc_scores)
            arcs = torch.flatten(arcs).type(torch.float)
            arc_loss = self.arc_high_loss(arc_scores, arcs)

        if rels.shape[0] == 0:
            return arc_loss
        else:
            rel_loss = self.rel_criterion(rel_scores, rels)
            return arc_loss + rel_loss

    @staticmethod
    def decode(arc_scores, rel_scores, mask):
        arc_preds = eisner(arc_scores, mask)
        rel_preds = rel_scores.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    def convert_examples_to_features(self, examples):

        features = []

        length_list = []
        tokens_list = []
        head_idx_list = []
        labels_list = []
        valid_list = []
        label_mask_list = []
        punctuation_idx_list = []

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a
            labellist = example.label
            head_list = example.head
            tokens = []
            head_idx = []
            labels = []
            valid = []
            label_mask = []

            punctuation_idx = []

            if len(textlist) > self.max_seq_length - 2:
                textlist = textlist[:self.max_seq_length - 2]
                labellist = labellist[:self.max_seq_length - 2]
                head_list = head_list[:self.max_seq_length - 2]

            for i, word in enumerate(textlist):
                if ispunct(word):
                    punctuation_idx.append(i+1)
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        head_idx.append(head_list[i])
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            if len(tokens) > 300:
                continue
            length_list.append(len(tokens))
            tokens_list.append(tokens)
            head_idx_list.append(head_idx)
            labels_list.append(labels)
            valid_list.append(valid)
            label_mask_list.append(label_mask)
            punctuation_idx_list.append(punctuation_idx)

        label_len_list = [len(label) for label in labels_list]
        seq_pad_length = max(length_list) + 2
        label_pad_length = max(label_len_list) + 1

        for example, tokens, head_idxs, labels, valid, label_mask, punctuation_idx in \
                zip(examples,
                    tokens_list, head_idx_list, labels_list, valid_list, label_mask_list, punctuation_idx_list):

            ntokens = []
            segment_ids = []
            label_ids = []
            head_idx = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            head_idx.append(-1)
            label_ids.append(self.labelmap["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            for i in range(len(labels)):
                if labels[i] in self.labelmap:
                    label_ids.append(self.labelmap[labels[i]])
                else:
                    label_ids.append(self.labelmap['<UNK>'])
                head_idx.append(head_idxs[i])
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)

            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < seq_pad_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                valid.append(1)
            while len(label_ids) < label_pad_length:
                head_idx.append(-1)
                label_ids.append(0)
                label_mask.append(0)

            eval_mask = copy.deepcopy(label_mask)
            eval_mask[0] = 0
            # ignore all punctuation if not specified
            for idx in punctuation_idx:
                if idx < label_pad_length:
                    eval_mask[idx] = 0

            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(head_idx) == label_pad_length
            assert len(label_mask) == label_pad_length
            assert len(eval_mask) == label_pad_length

            ngram_ids = None
            ngram_positions_matrix = None
            ngram_lengths = None
            ngram_tuples = None
            ngram_seg_ids = None
            ngram_mask_array = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              head_idx=head_idx,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              eval_mask=eval_mask,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array,
                              ))
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_head_idx = torch.tensor([f.head_idx for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.bool)
        all_eval_mask_ids = torch.tensor([f.eval_mask for f in feature], dtype=torch.bool)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        head_idx = all_head_idx.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        eval_mask = all_eval_mask_ids.to(device)


        ngram_ids = None
        ngram_positions = None
        return input_ids, input_mask, l_mask, eval_mask, head_idx, label_ids, \
               ngram_ids, ngram_positions, segment_ids, valid_ids,


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, head=None, label=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.head = head
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, head_idx, label_id, valid_ids=None,
                 label_mask=None, eval_mask=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.head_idx = head_idx
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.eval_mask = eval_mask

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks



def readfile(filename):
    data = []
    sentence = []
    head = []
    label = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        bad_sent = False
        for line in lines:
            line = line.strip()
            if line == '' or re.match('\\s+', line):
                if not (len(sentence) == len(head) and len(head) == len(label)):
                    bad_sent = True
                if len(sentence) > 0 and not bad_sent:
                    data.append((sentence, head, label))
                sentence = []
                head = []
                label = []
                bad_sent = False
                continue
            splits = line.split('\t')
            if re.match('\\s+', splits[1]) or re.match('\\s+', splits[6]) or re.match('\\s+', splits[7]):
                bad_sent = True
                continue
            sentence.append(splits[1])
            head.append(int(splits[6]))
            label.append(splits[7])
        data.append((sentence, head, label))
    return data
