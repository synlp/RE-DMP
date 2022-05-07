from __future__ import absolute_import, division, print_function

import math

import numpy as np
import torch
from torch import nn
from modules import BertModel, ZenModel, BertTokenizer, Biaffine, MLP
from transformers_xlnet import XLNetModel, XLNetTokenizer
from util import ZenNgramDict
from re_helper import save_json, load_json
import subprocess
import os


DEFAULT_HPARA = {
    'max_seq_length': 508,
    'use_bert': False,
    'use_xlnet': False,
    'use_zen': False,
    'do_lower_case': False,
    'mlp_dropout': 0.33,
    'n_mlp': 200,
    'use_biaffine': True
}


class RelationExtractor(nn.Module):

    def __init__(self, labelmap, hpara, model_path, from_pretrained=True):
        super().__init__()
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']
        self.use_biaffine = hpara['use_biaffine']

        if hpara['use_zen']:
            raise ValueError()

        self.tokenizer = None
        self.bert = None
        self.xlnet = None
        self.zen = None
        self.zen_ngram_dict = None

        if self.hpara['use_bert']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.tokenizer.add_never_split_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
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
            self.tokenizer.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
            if from_pretrained:
                self.xlnet = XLNetModel.from_pretrained(model_path)
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
                self.xlnet = XLNetModel(config)
            hidden_size = self.xlnet.config.hidden_size
            self.dropout = nn.Dropout(self.xlnet.config.summary_last_dropout)

            self.xlnet.resize_token_embeddings(len(self.tokenizer))
        elif self.hpara['use_zen']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.zen_ngram_dict = ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = ZenModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        self.mlp_e1 = MLP(n_in=hidden_size, n_hidden=self.hpara['n_mlp'], dropout=self.hpara['mlp_dropout'])
        self.mlp_e2 = MLP(n_in=hidden_size, n_hidden=self.hpara['n_mlp'], dropout=self.hpara['mlp_dropout'])

        if self.use_biaffine:
            self.biaffine = Biaffine(n_in=self.hpara['n_mlp'], n_out=self.num_labels, bias_x=True, bias_y=True)
        else:
            self.linear = nn.Linear(self.hpara['n_mlp'] * 2, self.num_labels, bias=True)

        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                entity_mark=None, labels=None,
                input_ngram_ids=None, ngram_position_matrix=None,
                ):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.xlnet is not None:
            transformer_outputs = self.xlnet(input_ids, token_type_ids, attention_mask=attention_mask)
            sequence_output = transformer_outputs[0]
        elif self.zen is not None:
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()

        batch_size, _, feat_dim = sequence_output.shape

        e1_h = sequence_output[range(sequence_output.size(0)), entity_mark[:, 0]]
        e2_h = sequence_output[range(sequence_output.size(0)), entity_mark[:, 1]]

        tmp_e1_h = self.mlp_e1(e1_h)
        tmp_e2_h = self.mlp_e2(e2_h)

        if self.use_biaffine:
            logits = self.biaffine(tmp_e1_h, tmp_e2_h)
        else:
            tmp = torch.cat([tmp_e1_h, tmp_e2_h], dim=1)
            logits = self.linear(tmp)

        if labels is not None:
            loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_xlnet'] = args.use_xlnet
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['mlp_dropout'] = args.mlp_dropout
        hyper_parameters['n_mlp'] = args.n_mlp

        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    def save_model(self, output_model_dir, vocab_dir):
        best_eval_model_dir = os.path.join(output_model_dir, 'model')
        if not os.path.exists(best_eval_model_dir):
            os.makedirs(best_eval_model_dir)

        output_model_path = os.path.join(best_eval_model_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        output_tag_file = os.path.join(best_eval_model_dir, 'labelset.json')
        save_json(output_tag_file, self.labelmap)

        output_hpara_file = os.path.join(best_eval_model_dir, 'hpara.json')
        save_json(output_hpara_file, self.hpara)

        output_config_file = os.path.join(best_eval_model_dir, 'config.json')
        with open(output_config_file, "w", encoding='utf-8') as writer:
            if self.bert:
                writer.write(self.bert.config.to_json_string())
            elif self.xlnet:
                writer.write(self.xlnet.config.to_json_string())
            elif self.zen:
                writer.write(self.zen.config.to_json_string())
        output_bert_config_file = os.path.join(best_eval_model_dir, 'bert_config.json')
        command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
        subprocess.run(command, shell=True)

        if self.bert:
            vocab_name = 'vocab.txt'
        elif self.xlnet:
            vocab_name = 'spiece.model'
        elif self.zen:
            vocab_name = 'vocab.txt'
        else:
            raise ValueError()
        vocab_path = os.path.join(vocab_dir, vocab_name)
        command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(best_eval_model_dir, vocab_name))
        subprocess.run(command, shell=True)

    @classmethod
    def load_model(cls, model_path, device):
        tag_file = os.path.join(model_path, 'labelset.json')
        labelmap = load_json(tag_file)

        hpara_file = os.path.join(model_path, 'hpara.json')
        hpara = load_json(hpara_file)
        DEFAULT_HPARA.update(hpara)

        res = cls(labelmap=labelmap, hpara=DEFAULT_HPARA, model_path=model_path)
        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
        return res

    def load_data(self, data_path, do_predict=False):
        if not do_predict:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
        else:
            flag = 'predict'

        lines = readfile(data_path)

        examples = self.process_data(lines, flag)

        return examples

    @staticmethod
    def process_data(lines, flag):
        examples = []
        for i, (e1, e2, label, sentence) in enumerate(lines):
            guid = "%s-%s" % (flag, i)
            examples.append(InputExample(guid=guid, text_a=sentence, label=label, e1=e1, e2=e2))
        return examples

    def convert_examples_to_features(self, examples):

        features = []

        length_list = []
        input_tokens_list = []
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        entity_mark_list = []

        for (ex_index, example) in enumerate(examples):

            tokens = ["[CLS]"]
            entity_mark = [0, 0]
            # previous_id = 1
            for word in example.text_a.split():
                token = self.tokenizer.tokenize(word)
                if word == '<e1>':
                    entity_mark[0] = len(tokens)
                if word == '<e2>':
                    entity_mark[1] = len(tokens)
                # if word in ["</e1>"]:
                #     entity_mark[0] = previous_id
                # if word in ["</e2>"]:
                #     entity_mark[1] = previous_id
                tokens.extend(token)
                # previous_id = len(tokens) - len(token)

            if len(tokens) > 510:
                continue
            tokens.append("[SEP]")
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            length_list.append(len(input_ids))
            input_tokens_list.append(tokens)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            entity_mark_list.append(entity_mark)

        max_seq_length = max(length_list) + 2

        for indx, (example, tokens, input_ids, input_mask, segment_ids, entity_mark) in \
                enumerate(zip(examples, input_tokens_list, input_ids_list, input_mask_list,
                              segment_ids_list, entity_mark_list)):

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = self.labelmap[example.label] if example.label in self.labelmap else self.labelmap['<UNK>']

            assert not label_id == 0

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
                ngram_positions_matrix = np.zeros(shape=(max_seq_length, self.zen_ngram_dict.max_ngram_in_seq), dtype=np.int32)
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
                              entity_mark=entity_mark,
                              label_id=label_id,
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
        all_entity_mark = torch.tensor([f.entity_mark for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)

        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)

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

        return input_ids, input_mask, all_entity_mark, label_ids, \
               ngram_ids, ngram_positions, segment_ids


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, e1=None, e2=None):
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
        self.label = label
        self.e1 = e1
        self.e2 = e2


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, entity_mark, label_id,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.entity_mark = entity_mark
        self.label_id = label_id

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def readfile(filename):
    data = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            # line = line.strip()
            line = line.replace("<e1> </e1>", "<e1> [E] </e1>").replace("<e2> </e2>", "<e2> [E] </e2>")

            splits = line.split('\t')
            if len(splits) < 1:
                continue
            e1, e2, label, sentence = splits
            sentence = sentence.strip()

            for token in ['<e1>', '</e1>', '<e2>', '</e2>']:
                sentence = sentence.replace(token, ' ' + token + ' ').replace('  ', ' ').strip()

            if len(e1) == 0:
                e1 = "[E]"
            if len(e2) == 0:
                e2 = "[E]"

            e11_p = sentence.index("<e1>")  # the start position of entity1
            e12_p = sentence.index("</e1>")  # the end position of entity1
            e21_p = sentence.index("<e2>")  # the start position of entity2
            e22_p = sentence.index("</e2>")  # the end position of entity2

            if e1 in sentence[e11_p:e12_p] and e2 in sentence[e21_p:e22_p]:
                data.append(splits)
            elif e2 in sentence[e11_p:e12_p] and e1 in sentence[e21_p:e22_p]:
                splits[0] = e2
                splits[1] = e1
                data.append(splits)
            else:
                print("data format error: {}".format(line))

    return data
