from __future__ import absolute_import, division, print_function

import copy
import math

import numpy as np
import torch
from torch import nn
from modules import BertModel, ZenModel, BertTokenizer, Biaffine, MLP
from transformers_xlnet import XLNetModel, XLNetTokenizer
from util import eisner, ZenNgramDict, ispunct
import re
import subprocess
import os
from dep_helper import load_json, save_json


DEFAULT_HPARA = {
    'max_seq_length': 128,
    'use_bert': False,
    'use_xlnet': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_second_order': False,
    'use_third_order': False,
    'use_biaffine': False
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
        self.use_second_order = self.hpara['use_second_order']
        self.use_third_order = self.hpara['use_third_order']

        self.use_biaffine = self.hpara['use_biaffine']

        if hpara['use_zen']:
            raise ValueError()

        self.tokenizer = None
        self.bert = None
        self.transformer = None
        self.zen = None
        self.zen_ngram_dict = None

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
        elif self.hpara['use_zen']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.zen_ngram_dict = ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = ZenModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        if self.use_biaffine:
            self.mlp_arc_h = MLP(n_in=hidden_size,
                                 n_hidden=500,
                                 dropout=0.33)
            self.mlp_arc_d = MLP(n_in=hidden_size,
                                 n_hidden=500,
                                 dropout=0.33)
            self.mlp_rel_h = MLP(n_in=hidden_size,
                                 n_hidden=100,
                                 dropout=0.33)
            self.mlp_rel_d = MLP(n_in=hidden_size,
                                 n_hidden=100,
                                 dropout=0.33)

            self.arc_attn = Biaffine(n_in=500,
                                     bias_x=True,
                                     bias_y=False)
            self.rel_attn = Biaffine(n_in=100,
                                     n_out=self.num_labels,
                                     bias_x=True,
                                     bias_y=True)
        else:
            self.linear_arc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.rel_classifier_1 = nn.Linear(hidden_size, self.num_labels, bias=False)
            self.rel_classifier_2 = nn.Linear(hidden_size, self.num_labels, bias=False)
            self.bias = nn.Parameter(torch.tensor(self.num_labels, dtype=torch.float), requires_grad=True)
            nn.init.zeros_(self.bias)

        if self.use_second_order:
            self.second_labelmap = {'father': 1, 'sister': 2, 'daughter': 3}
            num_l = len(self.second_labelmap) + 1
            if self.use_biaffine:
                self.mlp_arc_h_2 = MLP(n_in=hidden_size, n_hidden=500, dropout=0.33)
                self.mlp_arc_d_2 = MLP(n_in=hidden_size, n_hidden=500, dropout=0.33)
                self.mlp_rel_h_2 = MLP(n_in=hidden_size, n_hidden=100, dropout=0.33)
                self.mlp_rel_d_2 = MLP(n_in=hidden_size, n_hidden=100, dropout=0.33)

                self.arc_attn_2 = Biaffine(n_in=500, bias_x=True, bias_y=False)
                self.rel_attn_2 = Biaffine(n_in=100, n_out=num_l, bias_x=True, bias_y=True)
            else:
                self.linear_second_arc = nn.Linear(hidden_size, hidden_size, bias=False)
                self.rel_second_classifier_1 = nn.Linear(hidden_size, num_l, bias=False)
                self.rel_second_classifier_2 = nn.Linear(hidden_size, num_l, bias=False)
                self.second_bias = nn.Parameter(torch.tensor(num_l, dtype=torch.float), requires_grad=True)
                nn.init.zeros_(self.second_bias)
            self.arc_sigmoid = nn.Sigmoid()
            self.arc_high_loss = nn.BCELoss()

        if self.use_third_order:
            self.third_labelmap = {'father': 1, 'elder_sister': 2, 'younger_sister': 3, 'daughter': 4}
            num_l = len(self.third_labelmap) + 1
            if self.use_biaffine:
                self.mlp_arc_h_3 = MLP(n_in=hidden_size, n_hidden=500, dropout=0.33)
                self.mlp_arc_d_3 = MLP(n_in=hidden_size, n_hidden=500, dropout=0.33)
                self.mlp_rel_h_3 = MLP(n_in=hidden_size, n_hidden=100, dropout=0.33)
                self.mlp_rel_d_3 = MLP(n_in=hidden_size, n_hidden=100, dropout=0.33)

                self.arc_attn_3 = Biaffine(n_in=500, bias_x=True, bias_y=False)
                self.rel_attn_3 = Biaffine(n_in=100, n_out=num_l, bias_x=True, bias_y=True)
            else:
                self.linear_third_arc = nn.Linear(hidden_size, hidden_size, bias=False)
                self.rel_third_classifier_1 = nn.Linear(hidden_size, num_l, bias=False)
                self.rel_third_classifier_2 = nn.Linear(hidden_size, num_l, bias=False)
                self.third_bias = nn.Parameter(torch.tensor(num_l, dtype=torch.float), requires_grad=True)
                nn.init.zeros_(self.third_bias)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None,
                attention_mask_label=None,
                input_ngram_ids=None, ngram_position_matrix=None,
                arcs=None, rels=None,
                second_rels=None,
                third_rels=None,
                ):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.transformer is not None:
            transformer_outputs = self.transformer(input_ids, token_type_ids, attention_mask=attention_mask)
            sequence_output = transformer_outputs[0]
        # elif self.zen is not None:
        #     sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
        #                                   ngram_position_matrix=ngram_position_matrix,
        #                                   token_type_ids=token_type_ids, attention_mask=attention_mask,
        #                                   output_all_encoded_layers=False)
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

        if self.use_biaffine:
            valid_output = self.dropout(valid_output)

            arc_h = self.mlp_arc_h(valid_output)
            arc_d = self.mlp_arc_d(valid_output)
            rel_h = self.mlp_rel_h(valid_output)
            rel_d = self.mlp_rel_d(valid_output)

            # get arc and rel scores from the bilinear attention
            # [batch_size, seq_len, seq_len]
            s_arc = self.arc_attn(arc_d, arc_h)
            # [batch_size, seq_len, seq_len, n_rels]
            s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~attention_mask_label.unsqueeze(1), float('-inf'))
        else:
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
            if self.use_second_order:
                if self.use_biaffine:
                    arc_h_2 = self.mlp_arc_h_2(valid_output)
                    arc_d_2 = self.mlp_arc_d_2(valid_output)
                    rel_h_2 = self.mlp_rel_h_2(valid_output)
                    rel_d_2 = self.mlp_rel_d_2(valid_output)

                    # get arc and rel scores from the bilinear attention
                    # [batch_size, seq_len, seq_len]
                    s_arc_second = self.arc_attn_2(arc_d_2, arc_h_2)
                    # [batch_size, seq_len, seq_len, n_rels]
                    s_rel_second = self.rel_attn_2(rel_d_2, rel_h_2).permute(0, 2, 3, 1)
                    s_arc_second.masked_fill_(~attention_mask_label.unsqueeze(1), float(-1e10))
                    s_arc_second = self.arc_sigmoid(s_arc_second)
                else:
                    tmp_arc_second = self.linear_second_arc(valid_output).permute(0, 2, 1)
                    s_arc_second = torch.bmm(valid_output, tmp_arc_second)
                    s_arc_second.masked_fill_(~attention_mask_label.unsqueeze(1), float(-1e10))
                    s_arc_second = self.arc_sigmoid(s_arc_second)

                    rel_second_1 = self.rel_second_classifier_1(valid_output)
                    rel_second_2 = self.rel_second_classifier_2(valid_output)
                    rel_second_1 = torch.stack([rel_second_1] * max_len, dim=1)
                    rel_second_2 = torch.stack([rel_second_2] * max_len, dim=2)
                    s_rel_second = rel_second_1 + rel_second_2 + self.second_bias

                if self.use_third_order:
                    if self.use_biaffine:
                        arc_h_3 = self.mlp_arc_h_3(valid_output)
                        arc_d_3 = self.mlp_arc_d_3(valid_output)
                        rel_h_3 = self.mlp_rel_h_3(valid_output)
                        rel_d_3 = self.mlp_rel_d_3(valid_output)

                        # get arc and rel scores from the bilinear attention
                        # [batch_size, seq_len, seq_len]
                        s_arc_third = self.arc_attn_3(arc_d_3, arc_h_3)
                        # [batch_size, seq_len, seq_len, n_rels]
                        s_rel_third = self.rel_attn_3(rel_d_3, rel_h_3).permute(0, 2, 3, 1)
                        s_arc_third.masked_fill_(~attention_mask_label.unsqueeze(1), float(-1e10))
                        s_arc_third = self.arc_sigmoid(s_arc_third)
                    else:
                        tmp_arc_third = self.linear_third_arc(valid_output).permute(0, 2, 1)
                        s_arc_third = torch.bmm(valid_output, tmp_arc_third)
                        s_arc_third.masked_fill_(~attention_mask_label.unsqueeze(1), float(-1e10))
                        s_arc_third = self.arc_sigmoid(s_arc_third)
                        rel_third_1 = self.rel_third_classifier_1(valid_output)
                        rel_third_2 = self.rel_third_classifier_2(valid_output)
                        rel_third_1 = torch.stack([rel_third_1] * max_len, dim=1)
                        rel_third_2 = torch.stack([rel_third_2] * max_len, dim=2)
                        s_rel_third = rel_third_1 + rel_third_2 + self.third_bias

                    # first order loss
                    attention_mask_label[:, 0] = 0
                    first_order_loss = self.get_loss(s_arc, s_rel, arcs, rels, attention_mask_label)

                    # second order loss
                    second_arcs = torch.clamp(second_rels, max=1)
                    second_order_loss = self.get_loss(s_arc_second, s_rel_second,
                                                      second_arcs, second_rels, attention_mask_label, order='second')

                    third_arcs = torch.clamp(third_rels, max=1)
                    third_order_loss = self.get_loss(s_arc_third, s_rel_third,
                                                     third_arcs, third_rels, attention_mask_label, order='third')
                    return first_order_loss + second_order_loss + third_order_loss
                else:
                    # first order loss
                    attention_mask_label[:, 0] = 0
                    first_order_loss = self.get_loss(s_arc, s_rel, arcs, rels, attention_mask_label)
                    # second order loss
                    second_arcs = torch.clamp(second_rels, max=1)
                    second_order_loss = self.get_loss(s_arc_second, s_rel_second,
                                                      second_arcs, second_rels, attention_mask_label, order='second')

                    return first_order_loss + second_order_loss
            else:
                attention_mask_label[:, 0] = 0
                return self.get_loss(s_arc, s_rel, arcs, rels, attention_mask_label)

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_xlnet'] = args.use_xlnet
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case

        hyper_parameters['use_second_order'] = args.use_second_order
        hyper_parameters['use_third_order'] = args.use_third_order

        hyper_parameters['use_biaffine'] = args.use_biaffine

        if hyper_parameters['use_third_order']:
            assert hyper_parameters['use_second_order'] == True

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
            elif self.zen:
                writer.write(self.zen.config.to_json_string())
        output_bert_config_file = os.path.join(output_dir, 'bert_config.json')
        command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
        subprocess.run(command, shell=True)

        if self.bert:
            vocab_name = 'vocab.txt'
        elif self.transformer:
            vocab_name = 'spiece.model'
        elif self.zen:
            vocab_name = 'vocab.txt'
        else:
            raise ValueError()
        vocab_path = os.path.join(vocab_dir, vocab_name)
        command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(output_dir, vocab_name))
        subprocess.run(command, shell=True)

        if self.zen:
            ngram_name = 'ngram.txt'
            ngram_path = os.path.join(vocab_dir, ngram_name)
            command = 'cp ' + str(ngram_path) + ' ' + str(os.path.join(output_dir, ngram_name))
            subprocess.run(command, shell=True)

        if optimizer is not None:
            output_model_path = os.path.join(output_dir, 'optimizer.bin')
            torch.save(optimizer.state_dict(), output_model_path)

    def load_data(self, data_path):
        lines = readfile(data_path)
        data = []

        if self.use_second_order or self.use_third_order:
            for sentence, head, label, sec, third in lines:
                second_order_info, third_order_info = self.get_high_order_info(sentence, head)
                data.append((sentence, head, label, second_order_info, third_order_info))
            lines = data

        examples = self.process_data(lines)

        return examples

    @staticmethod
    def process_data(lines):

        examples = []
        for i, (sentence, head, label, second_order_info, third_order_info) in enumerate(lines):
            guid = "%s" % str(i)
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None, head=head,
                                         label=label,
                                         second_order_info=second_order_info,
                                         third_order_info=third_order_info))
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

    def get_high_order_info(self, sentence, head):
        node_info = [{'head': -1, 'tails': []} for _ in range(len(sentence) + 1)]
        for i, h_indx in enumerate(head):
            node_info[i+1]['head'] = h_indx
            node_info[h_indx]['tails'].append(i+1)

        second_order_info_list = []
        for i, node in enumerate(node_info):
            if i == 0:
                continue
            # find father nodes
            head_1 = node['head']
            if not head_1 == -1:
                head_2 = node_info[head_1]['head']
                if not head_2 == -1:
                    second_order_info_list.append((i, head_2, 'father'))
                # find sisters
                child_1_list = node_info[head_1]['tails']
                for j in child_1_list:
                    if i == j:
                        continue
                    second_order_info_list.append((i, j, 'sister'))

            # find daughters
            for tail in node['tails']:
                tail_1 = node_info[tail]
                for j in tail_1['tails']:
                    second_order_info_list.append((i, j, 'daughter'))

        if self.use_third_order:
            third_order_info_list = []
            for i, j, label in second_order_info_list:
                current_node = node_info[j]
                if label == 'father':
                    # find father
                    head_3 = current_node['head']
                    if not head_3 == -1:
                        third_order_info_list.append((i, head_3, 'father'))
                    # find elder sisters
                    children = current_node['tails']
                    for k in children:
                        if k == node_info[i]['head']:
                            continue
                        third_order_info_list.append((i, k, 'elder_sister'))
                # find younger sisters
                elif label == 'sister':
                    children = current_node['tails']
                    for k in children:
                        third_order_info_list.append((i, k, 'younger_sister'))
                # find daughters
                elif label == 'daughter':
                    children = current_node['tails']
                    for k in children:
                        third_order_info_list.append((i, k, 'daughter'))
        else:
            third_order_info_list = None
        return second_order_info_list, third_order_info_list

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

            # second order
            if self.use_second_order:
                rel_second = np.zeros((label_pad_length, label_pad_length), dtype=int)
                second_order_info = example.second_order_info
                bad_instance = False if len(second_order_info) > 0 else True
                for i, j, rel in second_order_info:
                    if i >= label_pad_length or j >= label_pad_length:
                        bad_instance = True
                        break
                    rel_second[i][j] = self.second_labelmap[rel]
                if bad_instance:
                    continue
            else:
                rel_second = None

            # third order
            if self.use_third_order:
                rel_third = np.zeros((label_pad_length, label_pad_length), dtype=int)
                third_order_info = example.third_order_info
                bad_instance = False
                for i, j, rel in third_order_info:
                    if i >= label_pad_length or j >= label_pad_length:
                        bad_instance = True
                        break
                    rel_third[i][j] = self.third_labelmap[rel]
                if bad_instance:
                    continue
            else:
                rel_third = None

            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(head_idx) == label_pad_length
            assert len(label_mask) == label_pad_length
            assert len(eval_mask) == label_pad_length

            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                max_gram_n = self.zen_ngram_dict.max_ngram_len

                for p in range(2, max_gram_n):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment,
                                                  self.zen_ngram_dict.ngram_to_freq_dict[character_segment]])

                ngram_matches = sorted(ngram_matches, key=lambda s: s[-1], reverse=True)

                max_ngram_in_seq_proportion = math.ceil((len(tokens) / self.max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(seq_pad_length, self.zen_ngram_dict.max_ngram_in_seq), dtype=np.int32)
                for i in range(len(ngram_ids)):
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
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
                              rel_second=rel_second,
                              rel_third=rel_third,
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

        if self.use_second_order:
            all_rel_second = torch.tensor([f.rel_second for f in feature], dtype=torch.long)
            rel_second = all_rel_second.to(device)
        else:
            rel_second = None

        if self.use_third_order:
            all_rel_third = torch.tensor([f.rel_third for f in feature], dtype=torch.long)
            rel_third = all_rel_third.to(device)
        else:
            rel_third = None

        if self.zen is not None:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)
            # all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
            # all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
            # all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None
        return input_ids, input_mask, l_mask, eval_mask, head_idx, label_ids, \
               ngram_ids, ngram_positions, segment_ids, valid_ids, \
               rel_second, rel_third


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, head=None, label=None,
                 second_order_info=None,
                 third_order_info=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.head = head
        self.label = label
        self.second_order_info = second_order_info
        self.third_order_info = third_order_info


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, head_idx, label_id, valid_ids=None,
                 label_mask=None, eval_mask=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None,
                 rel_second=None, rel_third=None):
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

        self.rel_second = rel_second
        self.rel_third = rel_third


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
                    data.append((sentence, head, label, None, None))
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
        data.append((sentence, head, label, None, None))
    return data
