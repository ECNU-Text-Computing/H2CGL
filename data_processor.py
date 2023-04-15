import argparse
import datetime
import json
import math
import os
import random
import re
from collections import defaultdict, Counter

import dgl
import joblib
import numpy as np
import pandas as pd
import scipy as sp
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer, BertModel
from torchtext.vocab import build_vocab_from_iterator, Vectors, vocab

from our_models.TSGCN import get_paper_ego_subgraph
from utilis.scripts import get_configs, IndexDict, add_new_elements
from utilis.bertwhitening_utils import input_to_vec, compute_kernel_bias, transform_and_normalize, inputs_to_vec
import networkx as nx
import pickle as pkl
# tokenizer = get_tokenizer('basic_english')
UNK, PAD, SEP = '[UNK]', '[PAD]', '[SEP]'


class DataProcessor:
    def __init__(self, data_source, max_len=256, seed=123, norm=False, log=False, time=None, model_config=None):
        print('Init...')
        self.data_root = './data/'
        self.data_source = data_source
        self.seed = int(seed)
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.mean = 0
        self.std = 1
        self.norm = norm
        self.log = log
        self.time = time
        self.config = model_config if model_config else dict()
        self.batch_graph = self.config.get('batch_graph', False)
        self.selected_ids = None
        print(self.time)
        # if self.data_source == 'pubmed':
        #     self.data_cat_path = self.data_root + self.data_source + '/'
        # elif self.data_source == 's2orc':
        #     self.data_cat_path = self.data_root + self.data_source + '/'
        # elif self.data_source == 'mag':
        #     self.data_cat_path = self.data_root + self.data_source + '/'
        # elif self.data_source == 'dblp':
        #     self.data_cat_path = self.data_root + self.data_source + '/'
        # elif self.data_source == 'sdblp':
        #     self.data_cat_path = self.data_root + self.data_source + '/'
        if self.data_source.endswith('pubmed'):
            self.data_cat_path = self.data_root + self.data_source + '/'
        elif self.data_source == 's2orc':
            self.data_cat_path = self.data_root + self.data_source + '/'
        elif self.data_source == 'mag':
            self.data_cat_path = self.data_root + self.data_source + '/'
        elif self.data_source.endswith('dblp'):
            self.data_cat_path = self.data_root + self.data_source + '/'

    def split_data(self, rate=0.8, fixed_num=None, shuffle=True, by='normal', time=None, cut_time=2015):
        if self.time:
            time = self.time
        all_values = json.load(open(self.data_cat_path + 'sample_citation_accum.json'))
        print(len(all_values))
        all_ids = list(all_values.keys())
        if self.data_source.endswith('pubmed'):
            info_dict = json.load(open(self.data_cat_path + 'sample_info_dict.json'))
            time_dict = dict(map(lambda x: (x[0], x[1]['pub_date']['year']), info_dict.items()))
            title_dict = dict(map(lambda x: (x[0], x[1]['title']), info_dict.items()))
            abs_dict = dict(map(lambda x: (x[0], x[1]['abstract']), info_dict.items()))
        elif self.data_source.endswith('dblp'):
            info_dict = json.load(open(self.data_cat_path + 'sample_info_dict.json'))
            time_dict = dict(map(lambda x: (x[0], x[1]['year']), info_dict.items()))
            title_dict = dict(map(lambda x: (x[0], x[1]['title']), info_dict.items()))
            del info_dict
            abs_dict = json.load(open(self.data_cat_path + 'sample_abstract_dict.json'))
        # elif self.data_source == 'sdblp':
        #     info_dict = json.load(open(self.data_cat_path + 'sample_info_dict.json'))
        #     time_dict = dict(map(lambda x: (x[0], x[1]['year']), info_dict.items()))
        #     title_dict = dict(map(lambda x: (x[0], x[1]['title']), info_dict.items()))
        #     del info_dict
        #     abs_dict = json.load(open(self.data_cat_path + 'sample_abstract_dict.json'))

        if by == 'time':
            # as by time, no rate and fixed_num, but use year
            # here only input year < time, but should know the difference
            train_time, val_time, test_time = time
            print('time:', time)
            print('cut time:', cut_time)
            test_ids = [key for key in all_values if int(time_dict[key]) <= test_time]
            print('test samples:', len(test_ids))
            test_idx = test_time + 5 - cut_time - 1
            print('test_idx', test_idx)
            test_values = list(map(lambda x: all_values[x][1][test_idx] - all_values[x][0][test_idx], test_ids))
            key_ids = set(map(lambda x: x[0],
                              filter(lambda x: x[1] >= self.config['cut_threshold'][-1], zip(test_ids, test_values))))
            selected_ids = list(key_ids)
            print('key_ids', len(key_ids))
            # count = 0
            random.seed(self.seed)
            random.shuffle(test_ids)
            for paper in test_ids:
                if paper not in key_ids:
                    selected_ids.append(paper)
                    # count += 1
                if len(selected_ids) >= fixed_num:
                    break

            all_values = dict([(paper, all_values[paper]) for paper in selected_ids])

            # train_ids = [key for key in all_values if int(info_dict[key]['pub_time']['year']) <= train_time]
            # print(len(train_ids))
            # val_ids = [key for key in all_values if int(info_dict[key]['pub_time']['year']) <= val_time]
            # print(len(val_ids))
            # test_ids = [key for key in all_values if int(info_dict[key]['pub_time']['year']) <= test_time]
            # print(len(test_ids))
            train_ids = [key for key in all_values if int(time_dict[key]) <= train_time]
            print('train samples:', len(train_ids))
            val_ids = [key for key in all_values if int(time_dict[key]) <= val_time]
            print('val samples:', len(val_ids))
            test_ids = [key for key in all_values if int(time_dict[key]) <= test_time]
            print('test samples:', len(test_ids))

            # here choose the cur + 5 as predict value 2015?
            # will be changed in the future
            # [2011, 2012, 2013, 2014, 2015], [2016, 2017, 2018, 2019, 2020]
            train_idx = train_time + 5 - cut_time - 1
            train_values = list(map(lambda x: all_values[x][1][train_idx] - all_values[x][0][train_idx], train_ids))
            val_idx = val_time + 5 - cut_time - 1
            val_values = list(map(lambda x: all_values[x][1][val_idx] - all_values[x][0][val_idx], val_ids))
            test_idx = test_time + 5 - cut_time - 1
            test_values = list(map(lambda x: all_values[x][1][test_idx] - all_values[x][0][test_idx], test_ids))
            print('time:', train_time, val_time, test_time)
            print('idx:', train_idx, val_idx, test_idx)

        else:
            if shuffle:
                random.seed(self.seed)
                print('data_processor seed', self.seed)
                random.shuffle(all_ids)

            total_count = len(all_ids)
            train_ids = all_ids[:int(total_count * rate)]
            val_ids = all_ids[int(total_count * rate): int(total_count * ((1 - rate) / 2 + rate))]
            test_ids = all_ids[int(total_count * ((1 - rate) / 2 + rate)):]

            train_values = list(map(lambda x: all_values[x], train_ids))
            val_values = list(map(lambda x: all_values[x], val_ids))
            test_values = list(map(lambda x: all_values[x], test_ids))

        train_contents = list(
            map(lambda x: re.sub('\s+', ' ', str(title_dict[x]) + '. ' + abs_dict[x]), train_ids))
        val_contents = list(
            map(lambda x: re.sub('\s+', ' ', str(title_dict[x]) + '. ' + abs_dict[x]), val_ids))
        test_contents = list(
            map(lambda x: re.sub('\s+', ' ', str(title_dict[x]) + '. ' + abs_dict[x]), test_ids))

        train_times = list(map(lambda x: int(time_dict[x]), train_ids))
        val_times = list(map(lambda x: int(time_dict[x]), val_ids))
        test_times = list(map(lambda x: int(time_dict[x]), test_ids))

        cut_data = {
            'train': [train_ids, train_values, train_contents, train_times],
            'val': [val_ids, val_values, val_contents, val_times],
            'test': [test_ids, test_values, test_contents, test_times],
        }

        torch.save(cut_data, self.data_cat_path + 'split_data')

    def show_graph_info(self, graph_name='graph_sample_feature_vector'):
        all_values = json.load(open(self.data_cat_path + 'sample_citation_accum.json'))
        print(len(all_values))
        all_ids = list(all_values.keys())
        trans_dict = json.load(open(self.data_cat_path + 'sample_node_trans.json'))
        all_papers = [trans_dict['paper'][paper] for paper in all_ids]
        graph = torch.load(self.data_cat_path + graph_name)
        all_refs = graph.in_degrees(torch.tensor(all_papers), etype='is cited by').numpy().tolist()
        all_cites = graph.in_degrees(torch.tensor(all_papers), etype='cites').numpy().tolist()
        all_times = graph.nodes['paper'].data['time'][all_papers, :].squeeze(dim=-1).numpy().tolist()
        df_count = pd.DataFrame(index=all_papers, data={'ref': all_refs, 'cite': all_cites, 'time': all_times})
        print(df_count.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]))
        df_count = df_count[df_count['time'] <= 2011]
        print(df_count.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]))

    def get_data(self, phases=None, selected_ids=None):
        cur_data = torch.load(self.data_cat_path + 'split_data')
        cur_phases = list(cur_data.keys())
        if phases is not None:
            for phase in cur_phases:
                if phase not in phases:
                    del cur_data[phase]
                # print(len(cur_data[phase]))
                # print(len(list(zip(*zip(*cur_data[phase])))))
                # print(len([*list(zip(cur_data[phase]))]))
        if selected_ids is not None:
            for phase in cur_data:
                temp_data = list(zip(*filter(lambda x: x[0] in selected_ids, zip(*cur_data[phase]))))
                cur_data[phase] = temp_data
                print('{}:'.format(phase), len(cur_data[phase][0]))
        return cur_data

    def get_selected_ids(self):
        df = pd.read_csv('./results/test/{}/hard_ones.csv'.format(self.data_source), index_col=0)
        self.selected_ids = set(list(df.index.astype(str)))
        return self.selected_ids

    def add_attrs(self):
        data = torch.load(self.data_cat_path + 'split_data')
        info = json.load(open(self.data_cat_path + 'sample_info_dict.json', 'r'))
        for phase in data:
            if len(data[phase]) > 3:
                continue
            if self.data_source == 'pubmed':
                pub_time = [int(info[paper]['pub_time']['year']) for paper in data[phase][0]]
            elif self.data_source in ['dblp', 'sdblp']:
                pub_time = [int(info[paper]['year']) for paper in data[phase][0]]
            # print(pub_time)
            data[phase].append(pub_time)
        torch.save(data, self.data_cat_path + 'split_data')

    def get_tokenizer(self, tokenizer_type='basic', tokenizer_path=None):
        if tokenizer_type == 'bert':
            self.tokenizer = CustomBertTokenizer(max_len=self.max_len, bert_path=tokenizer_path,
                                                 data_path=self.data_cat_path)
        elif tokenizer_type == 'glove':
            self.tokenizer = VectorTokenizer(max_len=self.max_len, vector_path=tokenizer_path,
                                             data_path=self.data_cat_path, name='glove')
        else:
            self.tokenizer = BasicTokenizer(max_len=self.max_len, data_path=self.data_cat_path)

        data = torch.load(self.data_cat_path + 'split_data')
        self.tokenizer.load_vocab(data['train'][2], seed=self.seed)
        return self.tokenizer

    def get_dataloader(self, batch_size=32, num_workers=0, graph_dict=None):
        print('workers', num_workers)
        # data = torch.load(self.data_cat_path + 'split_data')
        # self.data = data

        phases = ['train', 'val', 'test']
        if graph_dict:
            phases = [phase for phase in phases if phase in graph_dict['data']]
        data = self.get_data(phases, self.selected_ids)
        # all_last_values = map(lambda x: x[1][-1], data['train'][1])
        if self.norm:
            # all_train_values = np.sum(list(map(lambda x: [num for num in (x[0] + x[1]) if num >= 0], data['train'][1])))
            all_train_values = data['train'][1]
            self.mean = np.mean(all_train_values)
            self.std = np.std(all_train_values)
            print(self.mean)
            print(self.std)
        self.graph_dict = graph_dict
        collate_batch_method = self.collate_batch
        if self.batch_graph == 'ego':
            # collate_batch_method = {phase: self.graph_method_map(phase) for phase in phases}
            collate_batch_method = {phase: ego_collate_batch(self, phase) for phase in phases}
        elif self.batch_graph == 'emb':
            collate_batch_method = self.emb_collate_batch
            paper_idx_trans = dict(zip(self.graph_dict['node_trans']['paper'],
                                       range(len(self.graph_dict['node_trans']['paper']))))
            print(self.data_cat_path + self.config['emb_name'] + '_embs')
            all_embs = joblib.load(self.data_cat_path + self.config['emb_name'] + '_embs')
            print(len(all_embs))
            for phase in phases:
                paper_ids = data[phase][0]
                embs = [all_embs[paper_idx_trans[paper]] for paper in paper_ids]
                print(len(embs))
                data[phase].append(embs)
                # if phase == 'train':
                #     self.embs = torch.from_numpy(np.array(embs))

        if type(collate_batch_method) != dict:
            collate_batch_method = {phase: collate_batch_method for phase in phases}

        self.dataloaders = []
        for phase in phases:
            self.dataloaders.append(CustomDataLoader(dataset=list(zip(*data[phase])), batch_size=batch_size,
                                                     shuffle=True, num_workers=num_workers,
                                                     collate_fn=collate_batch_method[phase],
                                                     mean=self.mean, std=self.std, log=self.log, phase=phase))
        while len(self.dataloaders) < 3:
            self.dataloaders.append(None)
        return self.dataloaders

    def collate_batch(self, batch):
        values_list, content_list = [], []
        # inputs_list, valid_lens = [], []
        length_list = []
        mask_list = []
        ids_list = []
        time_list = []
        # print(batch)
        for (_ids, _values, _contents, _time) in batch:
            # processed_content, seq_len, mask = self.text_pipeline(_content)
            # # print(_label)
            processed_content, seq_len, mask = self.tokenizer.encode(_contents)
            # values_list.append(_values)
            # inputs, valid_len = self.values_pipeline(_values)
            # values_list.append(self.label_pipeline(_values))
            values_list.append(_values)
            # inputs_list.append(inputs)
            # valid_lens.append(valid_len)
            content_list.append(processed_content)
            length_list.append(seq_len)
            mask_list.append(mask)
            # ids_list.append(int(_ids.strip()))
            ids_list.append(_ids.strip())
            time_list.append(_time)

        # content_list = torch.cat(content_list)
        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        # inputs_list = torch.tensor(inputs_list, dtype=torch.float32)
        # valid_lens = torch.tensor(valid_lens, dtype=torch.int8)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        # ids_list = torch.tensor(ids_list, dtype=torch.int64)
        # print(len(label_list))
        # content_list = [content_batch, inputs_list, valid_lens]
        time_list = torch.tensor(time_list, dtype=torch.long)
        content_list = content_batch
        return content_list, values_list, length_list, \
               mask_list, ids_list, time_list, None

    def emb_collate_batch(self, batch):
        values_list, emb_batch = [], []
        # inputs_list, valid_lens = [], []
        length_list = []
        mask_list = []
        ids_list = []
        time_list = []
        # print(batch)
        for (_ids, _values, _contents, _time, _embs) in batch:
            # processed_content, seq_len, mask = self.text_pipeline(_content)
            # # print(_label)
            # processed_content, seq_len, mask = self.tokenizer.encode(_contents)
            # values_list.append(_values)
            # inputs, valid_len = self.values_pipeline(_values)
            # values_list.append(self.label_pipeline(_values))
            values_list.append(_values)
            # inputs_list.append(inputs)
            # valid_lens.append(valid_len)
            emb_batch.append(_embs)
            length_list.append(0)
            mask_list.append(0)
            # ids_list.append(int(_ids.strip()))
            ids_list.append(_ids.strip())
            time_list.append(_time)

        # content_list = torch.cat(content_list)
        # content_batch = torch.tensor(content_list, dtype=torch.int64)
        emb_batch = torch.from_numpy(np.array(emb_batch))
        # print(emb_batch.shape)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        # inputs_list = torch.tensor(inputs_list, dtype=torch.float32)
        # valid_lens = torch.tensor(valid_lens, dtype=torch.int8)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        # ids_list = torch.tensor(ids_list, dtype=torch.int64)
        # print(len(label_list))
        # content_list = [content_batch, inputs_list, valid_lens]
        time_list = torch.tensor(time_list, dtype=torch.long)
        return emb_batch, values_list, length_list, \
               mask_list, ids_list, time_list, None

    def get_embs(self, tokenizer_type, tokenizer_path, name='graph_sample_feature', mode='vector',
                 split_abstract=False):
        print(name)
        print(tokenizer_path)
        print('graph mode:', mode)
        # graph = torch.load(self.data_cat_path + 'graph_sample')
        node_trans = json.load(open(self.data_cat_path + 'sample_node_trans.json', 'r'))
        paper_trans = node_trans['paper']
        node_ids = list(paper_trans.values())
        node_ids.sort()
        print(len(node_ids))
        index_trans = dict(zip(paper_trans.values(), paper_trans.keys()))

        if not split_abstract:
            info_dict = json.load(open(self.data_cat_path + 'sample_info_dict.json', 'r'))
            abstracts = list(map(lambda x: info_dict[index_trans[x]]['abstract'].strip(), node_ids))
            del info_dict
        else:
            abs_dict = json.load(open(self.data_cat_path + 'sample_abstract_dict.json', 'r'))
            abstracts = list(map(lambda x: abs_dict[index_trans[x]].strip(), node_ids))
            del abs_dict

        feature_list = []

        self.tokenizer = self.get_tokenizer(tokenizer_type, tokenizer_path)
        self.tokenizer.load_vocab()
        print('bert-whitening embeddings')
        # self.tokenizer = CustomBertTokenizer(max_len=self.max_len, bert_path=tokenizer_path)
        # self.tokenizer.load_vocab()
        print(self.tokenizer.tokenizer)

        if mode == 'bw':
            bert_model = BertModel.from_pretrained(tokenizer_path, return_dict=True, output_hidden_states=True).to(
                self.device)
        elif mode in ['sbw', 'sbert']:
            bert_model = SentenceTransformer(tokenizer_path)

        # first get kernel and bias
        # params = {'kernel': 0, 'bias': 0}
        all_embs = []

        # for abstract in tqdm(abstracts[:100]):
        #     processed_content, seq_len, mask = self.tokenizer.encode(abstract)
        #     tokens = torch.tensor(processed_content).unsqueeze(dim=0).to(self.device)
        #     mask = torch.tensor(mask).unsqueeze(dim=0).to(self.device)
        #     # output = bert_model(tokens, attention_mask=mask)
        #     output = input_to_vec([tokens, mask], bert_model, 'first_last_avg', seq_len)
        #     all_embs.append(output)
        batch_size = 64
        if mode == 'bw':
            count = 0
            temp_tokens, temp_masks, temp_lens = [], [], []
            for abstract in tqdm(abstracts):
                processed_content, seq_len, mask = self.tokenizer.encode(abstract)
                temp_tokens.append(processed_content)
                temp_masks.append(mask)
                temp_lens.append(seq_len)
                count += 1
                if ((count > 0) and (count % batch_size == 0)) | (count == len(abstracts)):
                    tokens = torch.tensor(temp_tokens, dtype=torch.long).to(self.device)
                    masks = torch.tensor(temp_masks, dtype=torch.long).to(self.device)
                    seq_lens = temp_lens
                    output = inputs_to_vec([tokens, masks], bert_model, 'first_last_avg', seq_lens)
                    # print(output.shape)
                    all_embs.append(output)
                    temp_tokens, temp_masks, temp_lens = [], [], []
        elif mode in ['sbw', 'sbert']:
            count = 0
            all_embs = []
            temp_list = []
            for abstract in tqdm(abstracts):
                # print(abstract)
                temp_list.append(abstract)
                count += 1
                if ((count > 0) and (count % batch_size == 0)) | (count == len(abstracts)):
                    output = bert_model.encode(temp_list)
                    # print(output.shape)
                    all_embs.append(output)
                    temp_list = []

        all_embs = np.array(all_embs)
        all_embs = np.concatenate(all_embs, axis=0)

        print(all_embs.shape)
        joblib.dump(all_embs, self.data_cat_path + name + '_' + mode + '_embs')

    def get_feature_graph(self, tokenizer_type, tokenizer_path, name='graph_sample_feature', mode='vector',
                          split_abstract=False, time_range=(2001, 2015)):
        print(name)
        print(tokenizer_path)
        print('graph mode:', mode)
        graph = torch.load(self.data_cat_path + 'graph_sample')
        node_trans = json.load(open(self.data_cat_path + 'sample_node_trans.json', 'r'))
        paper_trans = node_trans['paper']
        node_ids = list(paper_trans.values())
        node_ids.sort()
        print(len(node_ids))
        index_trans = dict(zip(paper_trans.values(), paper_trans.keys()))

        if not split_abstract:
            info_dict = json.load(open(self.data_cat_path + 'sample_info_dict.json', 'r'))
            abstracts = list(map(lambda x: info_dict[index_trans[x]]['abstract'].strip(), node_ids))
            del info_dict
        else:
            abs_dict = json.load(open(self.data_cat_path + 'sample_abstract_dict.json', 'r'))
            abstracts = list(map(lambda x: abs_dict[index_trans[x]].strip(), node_ids))
            del abs_dict

        feature_list = []

        self.tokenizer = self.get_tokenizer(tokenizer_type, tokenizer_path)
        self.tokenizer.load_vocab()

        if mode == 'vector':
            # self.tokenizer = VectorTokenizer(max_len=self.max_len, vector_path=tokenizer_path,
            #                                  data_path=self.data_cat_path, name='glove')
            # self.tokenizer.load_vocab()
            print(self.tokenizer.vectors.shape)

            for abstract in tqdm(abstracts):
                processed_content, seq_len, mask = self.tokenizer.encode(abstract)
                node_embedding = self.tokenizer.vectors[processed_content][:seq_len].mean(dim=0, keepdim=True)
                if torch.isnan(node_embedding).sum().item() > 0:
                    print(abstract)
                    node_embedding = torch.zeros_like(node_embedding)
                feature_list.append(node_embedding)

        elif mode == 'bert':
            print('last 2nd avgpool')
            # self.tokenizer = CustomBertTokenizer(max_len=self.max_len, bert_path=tokenizer_path)
            # self.tokenizer.load_vocab()
            print(self.tokenizer.tokenizer)
            bert_model = BertModel.from_pretrained(tokenizer_path, return_dict=True, output_hidden_states=True).to(
                self.device)

            for abstract in tqdm(abstracts):
                processed_content, seq_len, mask = self.tokenizer.encode(abstract)
                tokens = torch.tensor(processed_content).unsqueeze(dim=0).to(self.device)
                mask = torch.tensor(mask).unsqueeze(dim=0).to(self.device)
                output = bert_model(tokens, attention_mask=mask)
                node_embedding = output['hidden_states'][-2][0, :seq_len].mean(dim=0, keepdim=True).detach().cpu()
                # print(node_embedding.shape)
                feature_list.append(node_embedding)

        elif mode == 'sbert':
            all_embs = joblib.load(self.data_cat_path + name + '_' + mode + '_embs')
            feature_list = torch.from_numpy(all_embs)

        elif mode.endswith('bw'):
            all_embs = joblib.load(self.data_cat_path + name + '_' + mode + '_embs')
            # kernel, bias = compute_kernel_bias([all_embs])
            # kernel = kernel[:, :300]  # 300 dim
            #
            # feature_list = torch.from_numpy(transform_and_normalize(all_embs, kernel, bias).astype(np.float32))
            # print(feature_list.shape)

            # then get the embs

        elif mode == 'token':
            # self.get_tokenizer(tokenizer_type, tokenizer_path)
            # self.tokenizer.load_vocab()
            seq_lens = []
            masks = []

            for abstract in tqdm(abstracts):
                processed_content, seq_len, mask = self.tokenizer.encode(abstract)
                processed_content = torch.tensor(processed_content).unsqueeze(dim=0)
                seq_len = torch.tensor(seq_len).unsqueeze(dim=0)
                mask = torch.tensor(mask).unsqueeze(dim=0)
                feature_list.append(processed_content)
                seq_lens.append(seq_len)
                masks.append(mask)

            #     graph.ndata['seq_len'] = torch.cat(seq_lens, dim=0)
            #     graph.ndata['mask'] = torch.cat(masks, dim=0)
            #
            # graph.ndata['h'] = torch.cat(feature_list, dim=0)
            # print(graph.ndata['h'].shape)
            # torch.save(graph, self.data_cat_path + name + '_' + mode)
            graph.nodes['paper'].data['seq_len'] = torch.cat(seq_lens, dim=0)
            graph.nodes['paper'].data['mask'] = torch.cat(masks, dim=0)

        if mode.endswith('bw'):
            # graph.nodes['paper'].data['h'] = feature_list
            print('not here')
        elif mode == 'sbert':
            graph.nodes['paper'].data['h'] = feature_list
        elif mode == 'random':
            paper_count = graph.nodes('paper').shape[0]
            graph.nodes['paper'].data['h'] = torch.randn(paper_count, 300, dtype=torch.float32)
        else:
            graph.nodes['paper'].data['h'] = torch.cat(feature_list, dim=0)
        print(graph.nodes['paper'].data['h'].shape)

        # author_src, author_dst = graph.edges(form='uv', etype='writes', order='srcdst')
        # journal_src, journal_dst = graph.edges(form='uv', etype='publishes', order='srcdst')
        author_src, author_dst = graph.edges(form='uv', etype='writes', order='eid')
        journal_src, journal_dst = graph.edges(form='uv', etype='publishes', order='eid')


        # torch.save(graph, self.data_cat_path + name + '_' + mode)
        joblib.dump(graph, self.data_cat_path + name + '_' + mode + '.job')

        for time_point in range(time_range[0], time_range[1] + 1):
            print('-' * 30 + str(time_point) + '-' * 30)
            sub_papers = graph.nodes('paper')[graph.nodes['paper'].data['time'].squeeze(dim=-1) <= time_point]
            print('paper', sub_papers.shape)
            selected_journal = list(
                set(journal_src[graph.edges['publishes'].data['time'].squeeze(dim=-1) <= time_point].numpy().tolist()))
            print('journal', len(selected_journal))
            selected_author = list(
                set(author_src[graph.edges['writes'].data['time'].squeeze(dim=-1) <= time_point].numpy().tolist()))
            print('author', len(selected_author))
            sub_nodes_dict = {
                'paper': sub_papers,
                'author': selected_author,
                'journal': selected_journal
            }
            sub_graph = dgl.node_subgraph(graph, sub_nodes_dict)

            sub_graph = dgl.remove_self_loop(sub_graph, 'is cited by')
            sub_graph = dgl.remove_self_loop(sub_graph, 'cites')

            if mode.endswith('bw'):
                cur_idx = sub_graph.nodes['paper'].data[dgl.NID]
                cur_embs = all_embs[cur_idx, :]
                print(cur_embs.shape)
                kernel, bias = compute_kernel_bias([cur_embs])
                kernel = kernel[:, :300]  # 300 dim

                feature_list = torch.from_numpy(transform_and_normalize(cur_embs, kernel, bias).astype(np.float32))
                print(feature_list.shape)
                sub_graph.nodes['paper'].data['h'] = feature_list

            if mode != 'token':
                emb_dim = sub_graph.nodes['paper'].data['h'].shape[1]
                author_features = []

                # print(sub_papers)
                # paper_oids = sub_graph.nodes['paper'].data[dgl.NID].numpy().tolist()
                # oid_cid_trans = dict(zip(paper_oids, range(len(paper_oids))))
                cur_author_src, cur_author_dst = sub_graph.edges(form='uv', etype='writes', order='eid')
                cur_journal_src, cur_journal_dst = sub_graph.edges(form='uv', etype='publishes', order='eid')
                # print(oid_cid_trans)
                for author in tqdm(sub_graph.nodes('author')):
                    temp_papers = cur_author_dst[cur_author_src == author]
                    if len(temp_papers) == 0:
                        raise Exception
                    # print(temp_papers)
                    author_features.append(
                        torch.mean(sub_graph.nodes['paper'].data['h'][temp_papers, :], dim=0, keepdim=True))
                if len(author_features) == 0:
                    sub_graph.nodes['author'].data['h'] = torch.zeros((0, emb_dim), dtype=torch.float32)
                else:
                    sub_graph.nodes['author'].data['h'] = torch.cat(author_features, dim=0)
                print(sub_graph.nodes['author'].data['h'].shape)

                journal_features = []
                for journal in tqdm(sub_graph.nodes('journal')):
                    # temp_papers = journal_dst[journal_src == journal]
                    # journal_features.append(
                    #     torch.mean(graph.nodes['paper'].data['h'][temp_papers], dim=0, keepdim=True))
                    # temp_papers = sub_graph.out_edges(journal, etype='publishes')[1]
                    # temp_papers = [oid_cid_trans[paper] for paper in journal_dst[journal_src == journal].numpy().tolist()]
                    temp_papers = cur_journal_dst[cur_journal_src == journal]
                    if len(temp_papers) == 0:
                        raise Exception
                    # print(temp_papers)
                    journal_features.append(
                        torch.mean(sub_graph.nodes['paper'].data['h'][temp_papers, :], dim=0, keepdim=True))
                if len(journal_features) == 0:
                    sub_graph.nodes['journal'].data['h'] = torch.zeros((0, emb_dim), dtype=torch.float32)
                else:
                    sub_graph.nodes['journal'].data['h'] = torch.cat(journal_features, dim=0)
                print(sub_graph.nodes['journal'].data['h'].shape)

                time_features = []
                cur_times = sorted(set(sub_graph.nodes['paper'].data['time'].squeeze(dim=-1).numpy().tolist()))
                for pub_time in tqdm(cur_times):
                    cur_index = np.argwhere(
                        sub_graph.nodes['paper'].data['time'].squeeze(dim=-1).numpy() == pub_time).squeeze(
                        axis=-1)
                    time_features.append(sub_graph.nodes['paper'].data['h'][cur_index].mean(dim=0, keepdim=True))
                time_features = torch.cat(time_features, dim=0)
                print(time_features.shape)
                sub_graph = add_new_elements(sub_graph, nodes={'time': len(cur_times)}, ndata={'time': {
                    'h': time_features,
                    '_ID': torch.tensor(cur_times, dtype=torch.long)
                }})

            print(sub_graph)
            # print(sub_graph.ntypes)
            # torch.save(sub_graph, self.data_cat_path + name + '_' + mode + '_' + str(time_point))
            joblib.dump(sub_graph, self.data_cat_path + name + '_' + mode + '_' + str(time_point) + '.job')

    def load_graphs(self, graph_name='graph_sample_feature', time_length=10, phases=('train', 'val', 'test')):
        print(graph_name)
        train_time, val_time, test_time = self.time
        print(self.time, time_length)
        phase_dict = {
            'train': list(range(train_time - (time_length - 1), train_time + 1)),
            'val': list(range(val_time - (time_length - 1), val_time + 1)),
            'test': list(range(test_time - (time_length - 1), test_time + 1))
        }
        phase_dict = dict(filter(lambda x: x[0] in phases, phase_dict.items()))
        print(phase_dict)
        # train_list = list(range(train_time - 9, train_time + 1))
        # val_list = list(range(val_time - 9, val_time + 1))
        # test_list = list(range(test_time - 9, test_time + 1))
        # all_time = set(phase_dict['train'] + phase_dict['val'] + phase_dict['test'])
        all_time = []
        for phase in phases:
            all_time += phase_dict[phase]
        all_time = set(all_time)

        graphs = {}
        for time in all_time:
            # graphs[time] = torch.load(self.data_cat_path + graph_name + '_' + str(time))
            graphs[time] = joblib.load(self.data_cat_path + graph_name + '_' + str(time) + '.job')

        data_dict = {phase: [graphs[time] for time in phase_dict[phase]] for phase in phase_dict}

        self.graph = {
            'data': data_dict,
            'node_trans': json.load(open(self.data_cat_path + 'sample_node_trans.json', 'r')),
            'time': {'train': train_time, 'val': val_time, 'test': test_time},
            'all_graphs': graphs,
            'time_length': time_length
        }
        return self.graph

    def get_time_dict(self):
        train_time, val_time, test_time = self.time
        return {'train': train_time, 'val': val_time, 'test': test_time}

    def values_pipeline(self, values):
        # inputs = values[0]
        # outputs = np.array(values[1])
        # inputs = list(filter(lambda x: x >= 0, inputs))
        # valid_len = len(inputs)
        # inputs = np.array(inputs + [0] * (len(outputs) - valid_len))
        # inputs = np.stack([inputs, outputs], axis=0)
        # if self.norm:
        #     inputs = (inputs - self.mean) / self.std
        # return inputs, valid_len
        return values


class CascadeDataProcessor(DataProcessor):
    def __init__(self, data_source, structure='individual', max_time=10, max_len=256, seed=123, norm=False, log=False,
                 time=None, model_config=None):
        super(CascadeDataProcessor, self).__init__(data_source, max_len, seed, norm, log, time, model_config)
        self.structure = structure
        self.max_time = max_time

    def get_cascade_graph(self, name='graph_sample_feature', mode='vector',
                          all_times=None, structure='individual', time_length=10):
        cur_start_time = datetime.datetime.now()
        phases = ['train', 'val', 'test']
        split_data = torch.load(self.data_cat_path + 'split_data')
        paper_trans = json.load(open(self.data_cat_path + 'sample_node_trans.json', 'r'))['paper']
        paper_reverse_trans = dict(zip(paper_trans.values(), paper_trans.keys()))
        total_count = 0
        times = {}
        for i in range(len(phases)):
            times[phases[i]] = all_times[i]
        print(times)

        if structure == 'individual':
            global_fw = open(self.data_cat_path + 'global_graph.txt', 'w+')
            global_graph = torch.load(self.data_cat_path + name + '_' + mode + '_' + str(times['train'] -
                                                                                         (time_length // 2)))
            global_oids = global_graph.nodes['paper'].data[dgl.NID].numpy().tolist()
            global_author_oids = global_graph.nodes['author'].data[dgl.NID].numpy().tolist()
            paper_src, author_dst = global_graph.edges(form='uv', etype='is writen by', order='srcdst')
            paper_src = [global_oids[x] for x in paper_src.numpy().tolist()]
            author_dst = [global_author_oids[x] for x in author_dst.numpy().tolist()]
            paper_author_dict = get_edge_dict(paper_src, author_dst)
            srcs, dsts = global_graph.edges(form='uv', etype='is cited by', order='srcdst')
            srcs = [global_oids[x] for x in srcs.numpy().tolist()]
            dsts = [global_oids[x] for x in dsts.numpy().tolist()]
            author_edges = defaultdict(int)
            all_authors = set(srcs + dsts)
            for src, dst in zip(srcs, dsts):
                src_authors = paper_author_dict[src]
                dst_authors = paper_author_dict[dst]
                for src_author in src_authors:
                    for dst_author in dst_authors:
                        if src_author != dst_author:
                            author_edges[(src_author, dst_author)] += 1
            # print(dict(filter(lambda x: x[1] > 1, author_edges.items())))
            for node in all_authors:
                edges = ['{}:{}'.format(key[1], author_edges[key]) for key in
                         list(filter(lambda x: x[0] == node, author_edges.keys()))]
                if edges:
                    line = str(node) + '\t\t' + '\t'.join(edges) + '\n'
                else:
                    line = str(node) + '\t\t' + 'null' + '\n'
                global_fw.write(line)

        phase_dict = defaultdict(int)
        group_dict = dict()

        for phase in phases:
            cur_graph = dgl.remove_self_loop(
                torch.load(self.data_cat_path + name + '_' + mode + '_' + str(times[phase])),
                etype='is cited by')
            cur_graph = dgl.node_type_subgraph(cur_graph, ntypes=['paper', 'author', 'journal'])
            # order alignment
            temp_papers = [paper_trans[paper] for paper in split_data[phase][0]]
            original_ids = cur_graph.nodes['paper'].data[dgl.NID].numpy().tolist()
            oid_cid_trans = dict(zip(original_ids, cur_graph.nodes('paper').numpy().tolist()))
            author_oids = cur_graph.nodes['author'].data[dgl.NID].numpy().tolist()
            # print(temp_starter)
            fw = open(self.data_cat_path + 'cascade_{}_{}.txt'.format(structure, phase), 'w+')

            paper_src, author_dst = cur_graph.edges(form='uv', etype='is writen by', order='srcdst')
            paper_src = [original_ids[x] for x in paper_src.numpy().tolist()]
            author_dst = [author_oids[x] for x in author_dst.numpy().tolist()]
            paper_author_dict = get_edge_dict(paper_src, author_dst)
            print('--paper_author_dict:', datetime.datetime.now() - cur_start_time, '--')

            if structure == 'individual':
                paper_src, paper_dst = cur_graph.edges(form='uv', etype='is cited by', order='srcdst')
                paper_src = [original_ids[x] for x in paper_src.numpy().tolist()]
                paper_dst = [original_ids[x] for x in paper_dst.numpy().tolist()]
                cite_edge_dict = get_edge_dict(paper_src, paper_dst)

                for paper in temp_papers:
                    # author as node
                    paper_dict = defaultdict(int)
                    cur_authors = paper_author_dict[paper]
                    # print(cur_authors)
                    cur_edges = cite_edge_dict[paper]
                    all_nodes = cur_authors.copy()
                    for cite_paper in cur_edges:
                        dsts = paper_author_dict[cite_paper]
                        all_nodes.extend(dsts)
                        for dst in dsts:
                            # print(cur_authors)
                            for src in cur_authors:
                                if src != dst:
                                    paper_dict[(src, dst)] += 1
                    cur_authors = [str(x) for x in cur_authors]
                    edges_list = ['{}:{}:{}'.format(key[0], key[1], paper_dict[key]) for key in paper_dict]
                    line = str('p' + paper_reverse_trans[paper]) + '\t' + ' '.join(cur_authors) + '\t' + str(
                        times[phase]) + '\t' + \
                           str(len(set(all_nodes))) + '\t' + ' '.join(edges_list) + '\n'

                    fw.write(line)
                    # total_count += 1

            elif structure == 'group':
                paper_src, paper_dst = cur_graph.edges(form='uv', etype='is cited by', order='srcdst')
                paper_src = [original_ids[x] for x in paper_src.numpy().tolist()]
                paper_dst = [original_ids[x] for x in paper_dst.numpy().tolist()]
                # cite_time = cur_graph.edges['is cited by'].data['time'][:, 0].numpy().tolist()
                paper_time = cur_graph.nodes['paper'].data['time'][:, 0].numpy().tolist()
                # cite_edge_dict = get_edge_dict(paper_src, paper_dst, cite_time)
                cite_edge_dict = get_edge_dict(paper_src, paper_dst)
                # group_dict = dict()
                # paper_graph = dgl.edge_type_subgraph(cur_graph, ['is cited by'])
                # nx_paper_graph = dgl.to_networkx(paper_graph, node_attrs=['time', '_ID'], edge_attrs=['time'])
                group_count = 0

                for paper in temp_papers:
                    # author group as node
                    paper_dict = defaultdict(int)
                    cur_authors = sorted(paper_author_dict[paper])
                    # cur_authors.sort()
                    cur_authors = str(cur_authors)
                    if cur_authors in group_dict:
                        cur_authors = group_dict[cur_authors]
                    else:
                        group_dict[cur_authors] = group_count
                        cur_authors = group_count
                        group_count += 1

                    # cur_edges = cite_edge_dict[paper]
                    all_nodes = []
                    # print('cur edges', cur_edges)
                    # shortest_paths = get_citation_cascade(cur_graph, original_ids.index(paper),
                    #                      [original_ids.index(paper)] +
                    #                      [original_ids.index(node) for node, _ in cite_edge_dict[paper]])
                    shortest_paths = get_citation_cascade(cur_graph, oid_cid_trans[paper],
                                                          [oid_cid_trans[node] for node in cite_edge_dict[paper] if
                                                           node != paper])
                    # shortest_paths = get_citation_cascade(cur_graph, oid_cid_trans[paper],
                    #                      [node for node in nx_paper_graph[oid_cid_trans[paper]]])

                    for path in shortest_paths:
                        # print('path', path)
                        cur_cite_time = paper_time[path[-1]]
                        path = [original_ids[node] for node in path]
                        groups = [str(sorted(paper_author_dict[paper])) for paper in path]
                        trans_groups = []
                        for group in groups:
                            if group in group_dict:
                                group = group_dict[group]
                            else:
                                group_dict[group] = group_count
                                group = group_count
                                group_count += 1
                            trans_groups.append(group)
                        all_nodes.append(trans_groups[-1])
                        paper_dict[tuple(trans_groups)] = cur_cite_time

                    path = sorted(paper_dict.items(), key=lambda x: x[1])
                    path = [','.join([str(node) for node in nodes]) + ':' + str(time) for nodes, time in path]

                    line = str('p' + paper_reverse_trans[paper]) + '\t' + str(cur_authors) + '\t' + str(
                        times[phase]) + '\t' + \
                           str(len(set(all_nodes))) + '\t' + ' '.join(path) + '\n'

                    fw.write(line)
                json.dump(group_dict, open(self.data_cat_path + 'cascade_group_dict.json', 'w+'))
                print('--phase:', datetime.datetime.now() - cur_start_time, '--')


            elif structure == 'paper':
                paper_src, paper_dst = cur_graph.edges(form='uv', etype='is cited by', order='srcdst')
                paper_src = [original_ids[x] for x in paper_src.numpy().tolist()]
                paper_dst = [original_ids[x] for x in paper_dst.numpy().tolist()]
                # cite_time = cur_graph.edges['is cited by'].data['time'][:, 0].numpy().tolist()
                paper_time = cur_graph.nodes['paper'].data['time'][:, 0].numpy().tolist()
                # cite_edge_dict = get_edge_dict(paper_src, paper_dst, cite_time)
                cite_edge_dict = get_edge_dict(paper_src, paper_dst)
                # group_dict = dict()
                # paper_graph = dgl.edge_type_subgraph(cur_graph, ['is cited by'])
                # nx_paper_graph = dgl.to_networkx(paper_graph, node_attrs=['time', '_ID'], edge_attrs=['time'])
                group_count = 0

                for paper in temp_papers:
                    # author group as node
                    paper_dict = defaultdict(int)
                    all_nodes = []
                    # print('cur edges', cur_edges)
                    # shortest_paths = get_citation_cascade(cur_graph, original_ids.index(paper),
                    #                      [original_ids.index(paper)] +
                    #                      [original_ids.index(node) for node, _ in cite_edge_dict[paper]])
                    shortest_paths = get_citation_cascade(cur_graph, oid_cid_trans[paper],
                                                          [oid_cid_trans[node] for node in cite_edge_dict[paper] if
                                                           node != paper])
                    # shortest_paths = get_citation_cascade(cur_graph, oid_cid_trans[paper],
                    #                      [node for node in nx_paper_graph[oid_cid_trans[paper]]])

                    for path in shortest_paths:
                        # print('path', path)
                        cur_cite_time = paper_time[path[-1]]
                        path = [original_ids[node] for node in path]
                        # groups = [str(sorted(paper_author_dict[paper])) for paper in path]
                        # trans_groups = []
                        # for group in groups:
                        #     if group in group_dict:
                        #         group = group_dict[group]
                        #     else:
                        #         group_dict[group] = group_count
                        #         group = group_count
                        #         group_count += 1
                        #     trans_groups.append(group)
                        all_nodes.append(path[-1])
                        paper_dict[tuple(path)] = cur_cite_time

                    path = sorted(paper_dict.items(), key=lambda x: x[1])
                    path = [','.join([str(node) for node in nodes]) + ':' + str(time) for nodes, time in path]

                    line = str(paper_reverse_trans[paper]) + '\t' + str(paper) + '\t' + str(
                        times[phase]) + '\t' + \
                           str(len(set(all_nodes))) + '\t' + ' '.join(path) + '\n'

                    fw.write(line)
                # json.dump(group_dict, open(self.data_cat_path + 'cascade_group_dict.json', 'w+'))
                print('--phase:', datetime.datetime.now() - cur_start_time, '--')

            elif structure == 'hdgnn':
                # p_p, p_a, a_p, p_v, v_p
                dir_path = self.data_cat_path + 'hdgnn/'
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)

                # paper citation
                paper_src, paper_dst = cur_graph.edges(form='uv', etype='is cited by', order='srcdst')
                paper_src = [original_ids[x] for x in paper_src.numpy().tolist()]
                paper_dst = [original_ids[x] for x in paper_dst.numpy().tolist()]
                # paper_src = [x for x in paper_src.numpy().tolist()]
                # paper_dst = [x for x in paper_dst.numpy().tolist()]
                get_simple_graph(dir_path + 'p_p_citation_list_{}.txt'.format(phase), paper_src, paper_dst)

                # paper, authors
                paper_src, author_dst = cur_graph.edges(form='uv', etype='is writen by', order='srcdst')
                paper_src = [original_ids[x] for x in paper_src.numpy().tolist()]
                author_dst = [author_oids[x] for x in author_dst.numpy().tolist()]
                # paper_src = [x for x in paper_src.numpy().tolist()]
                # author_dst = [x for x in author_dst.numpy().tolist()]
                get_simple_graph(dir_path + 'p_a_list_{}.txt'.format(phase), paper_src, author_dst)

                author_src, paper_dst = cur_graph.edges(form='uv', etype='writes', order='srcdst')
                paper_dst = [original_ids[x] for x in paper_dst.numpy().tolist()]
                author_src = [author_oids[x] for x in author_src.numpy().tolist()]
                # paper_dst = [x for x in paper_dst.numpy().tolist()]
                # author_src = [x for x in author_src.numpy().tolist()]
                get_simple_graph(dir_path + 'a_p_list_{}.txt'.format(phase), author_src, paper_dst)

                # paper, venues
                venue_oids = cur_graph.nodes['journal'].data[dgl.NID].numpy().tolist()
                paper_src, venue_dst = cur_graph.edges(form='uv', etype='is published by', order='srcdst')
                paper_src = [original_ids[x] for x in paper_src.numpy().tolist()]
                venue_dst = [venue_oids[x] for x in venue_dst.numpy().tolist()]
                # paper_src = [x for x in paper_src.numpy().tolist()]
                # venue_dst = [x for x in venue_dst.numpy().tolist()]
                get_simple_graph(dir_path + 'p_v_{}.txt'.format(phase), paper_src, venue_dst)

                venue_src, paper_dst = cur_graph.edges(form='uv', etype='publishes', order='srcdst')
                # print(cur_graph)
                paper_dst = [original_ids[x] for x in paper_dst.numpy().tolist()]
                venue_src = [venue_oids[x] for x in venue_src.numpy().tolist()]
                # paper_dst = [x for x in paper_dst.numpy().tolist()]
                # venue_src = [x for x in venue_src.numpy().tolist()]
                get_simple_graph(dir_path + 'v_p_list_{}.txt'.format(phase), venue_src, paper_dst)

                final_graph = torch.load(self.data_cat_path + 'graph_sample_feature_vector_2015')
                paper_embeds = final_graph.nodes['paper'].data['h'].numpy()
                print(paper_embeds.shape)
                pkl.dump(paper_embeds, open(dir_path + 'aps_title_emb.pkl', 'wb'))

    def split_labeled_data(self, name='graph_sample_feature', all_times=None, time_length=10):
        phases = ['train', 'val', 'test']
        times = {}
        for i in range(len(phases)):
            times[phases[i]] = all_times[i]
        print(times)

        for phase in phases:
            with open(self.data_cat_path + 'cascade_{}_{}.txt'.format(self.structure, phase), 'r') as fr:
                cur_data = [line.strip().split('\t') for line in fr]
            if phase == 'train':
                labeled_data = []
                unlabeled_data = []
                for data in cur_data:
                    paper = data[0]
                    data_time = int(data[2])
                    paths = data[-1].split(' ')
                    new_paths = []
                    root = data[1]
                    root_node, root_time = paths[0].split(':')
                    if len(root_node) != 1:
                        root_path = [path for path in paths if path.startswith(root + ':')][0]
                        root_time = int(root_path.split(':')[1])
                        paths = [root_path] + [path for path in paths if path != root_path]
                    else:
                        root_time = int(root_time)
                    # root_time = int(paths[0].split(':')[1])
                    for path in paths:
                        nodes, cur_time = path.split(':')
                        # cur_time = int(cur_time) - root_time
                        # cur_time = max(int(cur_time) - root_time, 0)
                        cur_time = max(0, int(cur_time) + 9 - data_time)
                        new_paths.append(':'.join([nodes, str(cur_time)]))
                    if root_time <= times['train'] - (time_length // 2):
                        labeled_data.append('\t'.join([paper, '\t'.join(new_paths), '0']))
                    else:
                        unlabeled_data.append('\t'.join([paper, '\t'.join(new_paths), '0']))
                with open(self.data_cat_path + 'ccgl/' + '{}.txt'.format(phase), 'w+') as fw:
                    for line in labeled_data:
                        fw.write(line + '\n')
                with open(self.data_cat_path + 'ccgl/' + 'unlabel.txt'.format(phase), 'w+') as fw:
                    for line in unlabeled_data:
                        fw.write(line + '\n')
            else:
                with open(self.data_cat_path + 'ccgl/' + '{}.txt'.format(phase), 'w+') as fw:
                    for data in cur_data:
                        paper = data[0]
                        data_time = int(data[2])
                        paths = data[-1].split(' ')
                        root = data[1]
                        new_paths = []
                        root_node, root_time = paths[0].split(':')
                        if len(root_node) != 1:
                            # print(root, paths)
                            root_path = [path for path in paths if path.startswith(root + ':')][0]
                            root_time = int(root_path.split(':')[1])
                            paths = [root_path] + [path for path in paths if path != root_path]
                        else:
                            root_time = int(root_time)
                        # root_time = int(paths[0].split(':')[1])
                        for path in paths:
                            nodes, cur_time = path.split(':')
                            # cur_time = int(cur_time) - root_time
                            # cur_time = max(int(cur_time) - root_time, 0)
                            cur_time = max(0, int(cur_time) + 9 - data_time)
                            new_paths.append(':'.join([nodes, str(cur_time)]))
                        fw.write('\t'.join([paper, '\t'.join(new_paths), '0']) + '\n')

    def get_tokenizer(self, tokenizer_type='vector', tokenizer_path=None):
        str_data = []
        # print(tokenizer_type)
        if tokenizer_type == 'vector':
            print('here')
            self.tokenizer = VectorTokenizer(max_len=self.max_len, vector_path=tokenizer_path,
                                             data_path=self.data_cat_path, name='node2vec')
            # data = torch.load(self.data_cat_path + 'split_data')
        elif tokenizer_type == 'group':
            # print('here')
            data = self.load_cascade_graph('group', ['train'])['train'][0]
            for walks in data:
                for walk, _ in walks:
                    str_data.append(' '.join(walk))
            # print(str_data)
            self.tokenizer = BasicTokenizer(max_len=self.max_len, data_path=self.data_cat_path)
        elif tokenizer_type == 'paper':
            # print('here')
            data = self.load_cascade_graph('paper', ['train'])['train'][0]
            for walks in data:
                for walk, _ in walks:
                    str_data.append(' '.join(walk))
            # print(str_data)
            self.tokenizer = BasicTokenizer(max_len=self.max_len, data_path=self.data_cat_path)
        else:
            self.tokenizer = BasicTokenizer(max_len=self.max_len, data_path=self.data_cat_path)
        self.tokenizer.load_vocab(str_data, seed=self.seed)
        return self.tokenizer

    def get_dataloader(self, batch_size=32, num_workers=0, graph_dict=None):
        data = torch.load(self.data_cat_path + 'split_data')
        phases = ['train', 'val', 'test']
        print(graph_dict)
        if graph_dict:
            phases = [phase for phase in phases if phase in graph_dict['data']]
        print('cascade phases:', phases)
        # [ids, values, contents]
        # all_last_values = map(lambda x: x[1][-1], data['train'][1])
        if self.norm:
            # all_train_values = np.sum(list(map(lambda x: [num for num in (x[0] + x[1]) if num >= 0], data['train'][1])))
            all_train_values = data['train'][1]
            self.mean = np.mean(all_train_values)
            self.std = np.std(all_train_values)
            print(self.mean)
            print(self.std)

        cascade_dict = self.load_cascade_graph(structure=self.structure, phases=phases)
        if self.structure == 'ccgl':
            for phase in phases:
                temp_list = []
                print(len(data[phase]))
                # phase_dict = {paper_id: values for (paper_id, values, contents, times) in zip(*data[phase])}
                phase_dict = dict(zip(data[phase][0], data[phase][1]))
                print(len(phase_dict.keys()))

                # for i in range(len(cascade_dict[phase])):
                #     if cascade_dict[phase][i][0] != data[phase][0][i]:
                #         print(cascade_dict[phase][i][0], data[phase][0][i])
                #         raise Exception
                #     else:
                #         print('all good!!!')

                for (paper_id, embed) in cascade_dict[phase]:
                    temp_list.append([paper_id, phase_dict[paper_id], embed])
                cascade_dict[phase] = list(zip(*temp_list))
                data[phase] = []
        elif self.structure in ['mucas', 'gtgcn']:
            for phase in phases:
                # X, PE, S, A ,T
                # cur_list = cascade_dict[phase]
                # id_list = data[phase][0]
                # print(len(cur_list))
                # print(len([cur_dict[paper] for paper in id_list]))
                # print([len(data) for data in [cur_dict[paper] for paper in id_list]])
                # cascade_dict[phase] = list(zip(*[cur_list[i][1:] for i in range(len(id_list))]))
                cascade_dict[phase] = list(zip(*cascade_dict[phase]))
                # print(len(cascade_dict[phase]))

        # print(.cascade_dict['train'])
        print('using', self.structure)
        collate_batch_method = None
        if self.structure == 'individual':
            collate_batch_method = self.individual_collate_batch
        elif self.structure in ['group', 'paper']:
            collate_batch_method = self.group_collate_batch
        if type(collate_batch_method) != dict:
            collate_batch_method = {phase: collate_batch_method for phase in phases}
        self.dataloaders = []
        for phase in phases:
            print(data.keys(), cascade_dict.keys())
            self.dataloaders.append(
                CustomDataLoader(dataset=list(zip(*(data[phase] + cascade_dict[phase]))), batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, collate_fn=collate_batch_method[phase],
                                 mean=self.mean, std=self.std, log=self.log, phase=phase))
        while len(self.dataloaders) < 3:
            self.dataloaders.append(None)
        return self.dataloaders

    def individual_collate_batch(self, batch):
        values_list, content_list = [], []
        # inputs_list, valid_lens = [], []
        length_list = []
        mask_list = []
        ids_list = []
        time_list = []
        # print(batch)
        for (_ids, _values, _contents, _time, _walks, _sz) in batch:
            # processed_content, seq_len, mask = self.tokenizer.encode(_contents)
            values_list.append(_values)
            # content_list.append(processed_content)
            processed_walks = []
            # print(len(_walks[0]))
            for walk in _walks:
                processed_walk, _, _ = self.tokenizer.encode(' '.join([str(x) for x in walk]))
                processed_walks.append(processed_walk)
            content_list.append(processed_walks)
            mask = [0]
            length_list.append(_sz)
            mask_list.append(mask)
            ids_list.append(_ids.strip())
            time_list.append(_time)
        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        time_list = torch.tensor(time_list, dtype=torch.long)
        content_list = content_batch
        return content_list, values_list, length_list, \
               mask_list, ids_list, time_list, None

    def group_collate_batch(self, batch):
        values_list, content_list = [], []
        # inputs_list, valid_lens = [], []
        length_list = []
        mask_list = []
        ids_list = []
        time_list = []
        rnn_list = []
        indicator_list = []
        pub_time_list = []
        # print(batch)
        count = 0
        for (_ids, _values, _contents, _time, _walks, _sz) in batch:
            # processed_content, seq_len, mask = self.tokenizer.encode(_contents)
            values_list.append(_values)
            # content_list.append(processed_content)
            # processed_walks = []
            # print(_walks)
            # print(_sz)
            # print(len(_walks))
            walk_count = 0
            for walk, time in _walks:
                # print(len(walks))
                # for (walk, time) in walks:
                time_index = torch.zeros((1, self.max_time), dtype=torch.int8)
                time_index[0, time] = 1
                rnn_index = torch.zeros((1, self.max_len), dtype=torch.int8)
                processed_walk, _, _ = self.tokenizer.encode(' '.join([str(x) for x in walk]))
                valid_len = np.sum(np.array(processed_walk) > self.tokenizer.vocab[PAD])
                rnn_index[0, valid_len - 1] = 1

                time_list.append(time_index)
                rnn_list.append(rnn_index)

                # processed_walks.append(processed_walk)
                content_list.append(processed_walk)
                for k in range(2 * self.config.get('hidden_size', 32)):
                    indicator_list.append([count, walk_count, k])
                walk_count += 1
            count += 1

            # content_list.append(processed_walks)
            mask = [0]
            length_list.append(_sz)
            mask_list.append(mask)
            ids_list.append(_ids.strip())
            pub_time_list.append(_time)
        # print(content_list)
        content_batch = torch.tensor(content_list, dtype=torch.int64)
        indicator_list = torch.tensor(indicator_list, dtype=torch.int64)
        time_list = torch.cat(time_list, dim=0)
        rnn_list = torch.cat(rnn_list, dim=0)
        pub_time_list = torch.tensor(pub_time_list, dtype=torch.long)

        values_list = torch.tensor(values_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        content_list = [content_batch, indicator_list, time_list, rnn_list]
        return content_list, values_list, length_list, \
               mask_list, ids_list, pub_time_list, None

    def load_cascade_graph(self, structure='individual', phases=('train', 'val', 'test')):
        # phases = ('train', 'val', 'test')
        result_dict = {}
        print(structure)

        if structure == 'individual':
            for phase in phases:
                graph_size = {}
                with open(self.data_cat_path + 'cascade_{}_{}.txt'.format(structure, phase)) as fr:
                    for line in fr:
                        parts = line.split('\t')
                        graph_size[parts[0]] = int(parts[3])

                temp_graph = {}
                with open(self.data_cat_path + 'random_walks_{}.txt'.format(phase), 'r') as f:
                    for line in f:
                        walks = line.strip().split('\t')
                        temp_graph[walks[0]] = []
                        for i in range(1, len(walks)):
                            temp_graph[walks[0]].append([int(x) for x in walks[i].split()])

                    x_data = []
                    sz_data = []
                    for key, graph in temp_graph.items():
                        temp = []
                        for walk in graph:
                            if len(walk) < self.max_len:
                                for i in range(self.max_len - len(walk)):
                                    # walk.append(-1)
                                    walk.append(PAD)
                            else:
                                walk = walk[:self.max_len]
                            temp.append(walk)
                        if len(graph) == 0:
                            # temp = [[-1] * 10] * 200
                            temp = [[PAD] * self.max_len] * 200
                        x_data.append(temp)
                        # y_data.append(np.log(y + 1.0) / np.log(2.0))
                        sz_data.append(graph_size[key])
                result_dict[phase] = [x_data, sz_data]

        elif structure in ['group', 'paper']:
            for phase in phases:
                graph_size = {}
                x_data = []
                sz_data = []
                with open(self.data_cat_path + 'cascade_{}_{}.txt'.format(structure, phase)) as fr:
                    for line in fr:
                        parts = line.split('\t')
                        graph_size[parts[0]] = int(parts[3])
                        #  (batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index)
                        # print(parts)
                        # time_range = [int(parts[2]) - 9, int(parts[2])]
                        walks = parts[-1].strip().split(' ')
                        dealt_walks = []
                        # walks = [([int(node) for node in walk.split(':')[0].split(',')], int(walk.split(':')[1]))
                        #          for walk in walks]
                        # walks = [(walk[0], walks[1] - time_range[1] + 10) for walk in walks]
                        # print(walks)
                        count = 0
                        for walk in walks:
                            walk = walk.split(':')
                            if count >= self.config['max_sequence']:
                                # print('too long')
                                break
                            temp_walk = [node.strip() for node in walk[0].split(',')]
                            walk_time = int(walk[1]) + self.max_time - 1 - int(parts[2])
                            walk_time = walk_time if walk_time > 0 else 0
                            if len(temp_walk) < self.max_len:
                                for i in range(self.max_len - len(temp_walk)):
                                    # walk.append(-1)
                                    temp_walk.append(PAD)
                            else:
                                temp_walk = temp_walk[:self.max_len]
                            dealt_walks.append((temp_walk, walk_time))
                            count += 1
                        # if len(dealt_walks) >=50:
                        #     print(len(dealt_walks))

                        x_data.append(dealt_walks)
                        # sz_data.append(int(parts[3]))
                        sz_data.append(len(dealt_walks))
                # print(sz_data)
                # print(x_data)
                print(len(x_data), len(sz_data))
                print('max size', max(sz_data))
                result_dict[phase] = [x_data, sz_data]

        elif structure == 'ccgl':
            for phase in phases:
                with open(self.data_cat_path + 'ccgl/{}.txt'.format(phase), 'r') as fr:
                    # paper_ids = [line.strip().split('\t')[0][1:] for line in fr]
                    # paper_ids = [line.strip().split('\t')[0] for line in fr]
                    paper_ids = [line.strip().split('\t')[0] for line in fr]
                paper_ids = [paper_id[1:] if paper_id[0] == 'p' else paper_id for paper_id in paper_ids]
                # embed_data = pkl.load(open(self.data_cat_path + 'ccgl/{}.pkl'.format(phase), 'rb'))[0]
                embed_data = joblib.load(self.data_cat_path + 'ccgl/{}.job'.format(phase))[0]
                if len(paper_ids) != len(embed_data):
                    print('invalid data!')
                    raise Exception
                result_dict[phase] = list(zip(paper_ids, embed_data))

        elif structure == 'mucas':
            for phase in phases:
                # phase_data = torch.load(self.data_cat_path + 'mucas_{}'.format(phase))
                phase_data = joblib.load(open(self.data_cat_path + 'mucas_{}.pkl'.format(phase), 'rb'))
                result_dict[phase] = phase_data

        elif structure == 'gtgcn':
            # for phase in phases:
            #     graph_size = {}
            #     x_data = []
            #     sz_data = []
            #     with open(self.data_cat_path + 'cascade_group_{}.txt'.format(phase)) as fr:
            #         for line in fr:
            #             parts = line.split('\t')
            #             graph_size[parts[0]] = int(parts[3])
            #             walks = parts[-1].strip().split(' ')
            #             # dealt_walks = []
            #             count = 0
            #             dealt_snapshots = [[] for i in range(self.max_time)]
            #             for walk in walks:
            #                 walk = walk.split(':')
            #                 # if count >= self.config['max_sequence']:
            #                 #     print('too long')
            #                 #     break
            #                 temp_walk = [node.strip() for node in walk[0].split(',')]
            #                 # 最近十年的切片而不是发布后10年的切片，可能会影响模型预测，需要注意一下
            #                 walk_time = int(walk[1]) + self.max_time - 1 - int(parts[2])
            #                 walk_time = walk_time if walk_time > 0 else 0
            #                 if walk_time < self.max_time:
            #                     dealt_snapshots[walk_time].append(temp_walk)
            #                 # dealt_walks.append((temp_walk, walk_time))
            #                 # count += 1
            #             # if len(dealt_walks) >=50:
            #             #     print(len(dealt_walks))
            #
            #             x_data.append(dealt_snapshots)
            #             # sz_data.append(int(parts[3]))
            #             # sz_data.append(len(dealt_walks))
            #     # print(sz_data)
            #     # print(x_data)
            #     # print(len(x_data), len(sz_data))
            #     # print('max size', max(sz_data))
            #     result_dict[phase] = [x_data]
            for phase in phases:
                # phase_data = torch.load(self.data_cat_path + 'mucas_{}'.format(phase))
                phase_data = joblib.load(open(self.data_cat_path + 'gtgcn_{}.pkl'.format(phase), 'rb'))
                result_dict[phase] = phase_data

        return result_dict


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=None,
                 mean=0, std=1, log=False, phase=None):
        super(CustomDataLoader, self).__init__(dataset=dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        self.mean = mean
        self.std = std
        self.log = log
        self.phase = phase


class BasicTokenizer:
    def __init__(self, max_len=256, data_path=None):
        self.tokenizer = get_tokenizer('basic_english')
        self.max_len = max_len
        self.vocab = None
        self.data_path = data_path
        self.vectors = None

    def yield_tokens(self, data_iter):
        for content in data_iter:
            yield self.tokenizer(content)

    def build_vocab(self, text_list, seed):
        self.vocab = build_vocab_from_iterator(self.yield_tokens(text_list), specials=[UNK, PAD])
        self.vocab.set_default_index(self.vocab[UNK])
        torch.save(self.vocab, self.data_path + 'vocab_{}'.format(seed))

    def load_vocab(self, text_list, seed):
        try:
            self.vocab = torch.load(self.data_path + 'vocab_{}'.format(seed))
        except Exception as e:
            print(e)
            self.build_vocab(text_list, seed)

    def encode(self, text):
        tokens = self.tokenizer(text)
        seq_len = len(tokens)
        if seq_len <= self.max_len:
            tokens += (self.max_len - seq_len) * [PAD]
        else:
            tokens = tokens[:self.max_len]
            seq_len = self.max_len
        ids = self.vocab(tokens)
        masks = [1] * seq_len + [0] * (self.max_len - seq_len)
        return ids, seq_len, masks


class CustomBertTokenizer(BasicTokenizer):
    def __init__(self, max_len=256, bert_path=None, data_path=None):
        super(CustomBertTokenizer, self).__init__(max_len)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def build_vocab(self, text_list=None, seed=None):
        self.vocab = {
            PAD: self.tokenizer.convert_tokens_to_ids([PAD])[0],
            UNK: self.tokenizer.convert_tokens_to_ids([UNK])[0],
            SEP: self.tokenizer.convert_tokens_to_ids([SEP])[0]
        }
        print('bert already have vocab')

    def load_vocab(self, text_list=None, seed=None):
        self.vocab = {
            PAD: self.tokenizer.convert_tokens_to_ids([PAD])[0],
            UNK: self.tokenizer.convert_tokens_to_ids([UNK])[0],
            SEP: self.tokenizer.convert_tokens_to_ids([SEP])[0]
        }
        print('bert already have vocab')

    def encode(self, text):
        result = self.tokenizer(text)
        result = self.tokenizer.pad(result, padding='max_length', max_length=self.max_len)
        ids = result['input_ids']
        mask = result['attention_mask']
        seq_len = sum(mask)
        # SEP_IDX = self.tokenizer.convert_tokens_to_ids([SEP])
        SEP_IDX = self.tokenizer.vocab[SEP]
        if seq_len > self.max_len:
            ids = ids[:self.max_len - 1] + [SEP_IDX]
            mask = mask[:self.max_len]
            seq_len = self.max_len
        return ids, seq_len, mask


class VectorTokenizer(BasicTokenizer):
    def __init__(self, max_len=256, vector_path=None, data_path=None, name=None):
        super(VectorTokenizer, self).__init__(max_len, data_path)
        self.vector_path = vector_path
        self.vectors = None
        self.name = name

    def build_vocab(self, text_list=None, seed=None):
        vec = Vectors(self.vector_path)
        self.vocab = vocab(vec.stoi, min_freq=0)
        self.vocab.append_token(UNK)
        self.vocab.append_token(PAD)
        self.vocab.set_default_index(self.vocab[UNK])
        unk_vec = torch.mean(vec.vectors, dim=0).unsqueeze(0)
        pad_vec = torch.zeros(vec.vectors.shape[1]).unsqueeze(0)
        self.vectors = torch.cat([vec.vectors, unk_vec, pad_vec])
        if self.name:
            torch.save(self.vocab, self.data_path + '{}_vocab'.format(self.name))
            torch.save(self.vectors, self.data_path + self.name)
        else:
            torch.save(self.vocab, self.data_path + 'vector_vocab')
            torch.save(self.vectors, self.data_path + 'vectors')

    def load_vocab(self, text_list=None, seed=None):
        try:
            if self.name:
                self.vocab = torch.load(self.data_path + '{}_vocab'.format(self.name))
                self.vectors = torch.load(self.data_path + self.name)
            else:
                self.vocab = torch.load(self.data_path + 'vector_vocab')
                self.vectors = torch.load(self.data_path + 'vectors')
        except Exception as e:
            print(e)
            self.build_vocab()


def make_data(data_source, config, seed=True, graph=None):
    dataProcessor = DataProcessor(data_source, seed=int(seed), model_config=config)
    # dataProcessor.split_data(config['rate'], config['fixed_num'])
    if os.path.exists(dataProcessor.data_cat_path + 'split_data'):
        print('split data already exist')
    else:
        dataProcessor.split_data(by='time', fixed_num=config['fixed_num'], time=config['time'],
                                 cut_time=config['cut_time'])
    dataProcessor.get_tokenizer(config['tokenizer_type'], config['tokenizer_path'])
    if graph:
        dataProcessor.get_feature_graph(config['tokenizer_type'], config['tokenizer_path'],
                                        mode=graph, split_abstract=config['split_abstract'],
                                        time_range=(config['time'][0] - config['time_length'] + 1, config['cut_time']))


def get_edge_dict(srcs, dsts, values=None):
    src_dst_dict = defaultdict(list)
    for i in range(len(srcs)):
        if values:
            src_dst_dict[srcs[i]].append((dsts[i], values[i]))
        else:
            src_dst_dict[srcs[i]].append(dsts[i])
    return src_dst_dict


def get_citation_cascade(graph, target, nodes):
    # print(nodes)
    # target_id = graph.nodes['paper'].data[dgl.NID][target].item()
    subgraph = dgl.node_subgraph(graph, {'paper': [target] + list(set(nodes))})
    src, dst = subgraph.edges(form='uv', etype='is cited by', order='srcdst')
    # print(src, dst)
    # special_index = np.intersect1d(np.argwhere(src.numpy() > 0)[:, 0], np.argwhere(dst.numpy() > 0)[:, 0]).tolist()
    # print(special_index)
    special_index = np.argwhere(src.numpy() > 0)[:, 0].tolist()
    src_s = src[special_index].numpy().tolist()
    dst_s = dst[special_index].numpy().tolist()
    nodes_counter = Counter(dst_s)
    oids = subgraph.nodes['paper'].data[dgl.NID].numpy().tolist()
    # print(nodes_counter)
    # target_time = subgraph.nodes['paper'].data['time'][0].item()

    dealt_src, dealt_dst = [], []
    for dst_node in dst.numpy().tolist():
        if (dst_node not in nodes_counter) and (dst_node != 0):
            dealt_src.append(0)
            dealt_dst.append(dst_node)
    add_nodes = [0] + dealt_dst.copy()

    for node, count in nodes_counter.items():
        if (count == 1) & (node != 0):
            dealt_dst.append(node)
            dealt_src.append(src_s[dst_s.index(node)])
        elif node != 0:
            # cur_time = subgraph.nodes['paper'].data['time'][node].item()
            cur_index = np.argwhere(np.array(dst_s) == node)[:, 0].tolist()
            src_nodes = np.array(src_s)[cur_index]
            time_value = subgraph.nodes['paper'].data['time'][src_nodes][:, 0].numpy()
            max_value = np.max(time_value)
            max_index = np.argwhere(time_value == max_value)[:, 0].tolist()
            final_nodes = set(src_nodes[max_index])
            # if cur_time == max_value:
            #     for src_node in final_nodes:
            #         dealt_dst.append(src_node)
            #         dealt_src.append(0)

            for src_node in final_nodes:
                dealt_dst.append(node)
                dealt_src.append(src_node)

    src_counter = Counter(src_s)
    edge_counter = Counter(list(zip(src_s, dst_s)) + list(zip(dst_s, src_s)))
    edge_counter = dict([(key, value) for key, value in edge_counter.items() if value > 1])
    # temp_src, temp_dst = dealt_src.copy(), dealt_dst.copy()
    for edge in edge_counter:
        add_nodes_set = set(add_nodes)
        if (edge[0] not in add_nodes_set) & (edge[1] not in add_nodes_set):
            if src_counter[edge[0]] == src_counter[edge[1]]:
                add_nodes.extend([edge[0], edge[1]])
                dealt_dst.append(edge[1])
                dealt_src.append(0)
                dealt_dst.append(edge[0])
                dealt_src.append(0)
            elif src_counter[edge[0]] < src_counter[edge[1]]:
                # add_nodes.append(edge[1])
                add_nodes.extend([edge[0], edge[1]])
                dealt_dst.append(edge[1])
                dealt_src.append(0)
            else:
                # add_nodes.append(edge[0])
                add_nodes.extend([edge[0], edge[1]])
                dealt_dst.append(edge[0])
                dealt_src.append(0)

    dealt_graph = nx.DiGraph()
    # all_nodes = subgraph.nodes('paper').numpy().tolist()
    all_nodes = [oids[node] for node in subgraph.nodes('paper').numpy().tolist()]
    # all_nodes = [oids[node] for node in sorted((set([0] + dst.numpy().tolist())))]

    # print('-' * 59)
    # # print(sorted(add_nodes))
    # print(subgraph.nodes('paper').numpy().tolist())
    # print(all_nodes)
    # print(subgraph.nodes['paper'].data['time'].numpy().tolist())
    # print(src, dst)
    # # print(temp_src, temp_dst)
    # print(dealt_src, dealt_dst)

    dealt_graph.add_nodes_from(all_nodes)
    dealt_src = [oids[node] for node in dealt_src]
    dealt_dst = [oids[node] for node in dealt_dst]
    dealt_graph.add_edges_from(set(zip(dealt_src, dealt_dst)))
    # print(dealt_graph)
    shortest_paths = [[target]]
    target_nodes = [node for node in all_nodes if node != target]
    # no cycles
    nc_nodes = [node for node in target_nodes if not nx.has_path(dealt_graph, source=target, target=node)]
    while len(nc_nodes) > 0:
        max_target_node = nc_nodes[np.argmax([src_counter[node] for node in nc_nodes])]
        dealt_graph.add_edge(target, max_target_node)
        nc_nodes = [node for node in nc_nodes if not nx.has_path(dealt_graph, source=target, target=node)]

    # for node in all_nodes[1:]:
    for node in target_nodes:
        # shortest_paths.extend([[oids[node] for node in p] for p in nx.all_shortest_paths(dealt_graph, source=all_nodes[0], target=node)])
        # shortest_paths.extend(
        #     [p for p in nx.all_shortest_paths(dealt_graph, source=all_nodes[0], target=node)])
        try:
            shortest_paths.extend(
                [p for p in nx.all_shortest_paths(dealt_graph, source=target, target=node)])
        except Exception as e:
            print(e)
            print(target)
            print(target, nodes)
            print(subgraph.nodes['paper'].data[dgl.NID].numpy().tolist())
            print(src, dst)
            print(dealt_src, dealt_dst)
            raise Exception
    # print(shortest_paths)
    # return dealt_src, dealt_dst
    return shortest_paths


def get_simple_graph(filename, srcs, dsts):
    fw = open(filename, 'w+')
    cur_src = -1
    for src, dst in zip(srcs, dsts):
        if src != dst:
            if cur_src == -1:
                fw.write('{}:{}'.format(str(src), str(dst)))
                print('start writing {}'.format(filename))
                cur_src = src
            elif cur_src != src:
                fw.write('\n')
                fw.write('{}:{}'.format(str(src), str(dst)))
                cur_src = src
            else:
                fw.write(',' + str(dst))

class ego_collate_batch:
    def __init__(self, dataProcessor, phase):
        self.phase = phase
        self.dataProcessor = dataProcessor

    def __call__(self, batch, *args, **kwargs):
        # for those that need dealing with graph in each batch
        values_list, content_list = [], []
        length_list = []
        mask_list = []
        ids_list = []
        time_list = []
        # print(batch)
        for (_ids, _values, _contents, _time) in batch:
            processed_content, seq_len, mask = self.dataProcessor.tokenizer.encode(_contents)
            values_list.append(_values)
            content_list.append(processed_content)
            length_list.append(seq_len)
            mask_list.append(mask)
            ids_list.append(_ids.strip())
            time_list.append(_time)
        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        time_list = torch.tensor(time_list, dtype=torch.long)
        content_list = content_batch

        paper_ids = [self.dataProcessor.graph_dict['node_trans']['paper'][paper] for paper in ids_list]
        # blocks = get_dealt_blocks(self.dataProcessor.config['model_name'], paper_ids, self.dataProcessor.config['hop'],
        #                           self.dataProcessor.graph_dict['data'][self.phase],
        #                           structure=self.dataProcessor.config['graph_structure'])
        subgraphs = [get_paper_ego_subgraph(paper_ids, hop=self.dataProcessor.config['hop'], graph=graph)
                     for graph in self.dataProcessor.graph_dict['data'][self.phase]]
        # pool = multiprocessing.Pool(len(self.dataProcessor.graph_dict['data'][self.phase]) // 2)
        # temp_mp = TM(paper_ids, self.dataProcessor.config['hop'])
        # subgraphs = pool.map(temp_mp.get_subgraph, self.dataProcessor.graph_dict['data'][self.phase])

        return content_list, values_list, length_list, \
               mask_list, ids_list, time_list, subgraphs


def show_hn(data_path='./data/sdblp/', train_time=2011):
    # hard negative
    print(train_time)
    ref_dict = json.load(open(data_path + 'sample_ref_dict.json'))
    cite_dict = defaultdict(list)
    for paper in ref_dict:
        for ref_paper in ref_dict[paper]:
            cite_dict[ref_paper].append(paper)
    json.dump(cite_dict, open(data_path + 'sample_cite_dict.json', 'w+'))

    samples = torch.load(data_path + 'split_data')
    print(samples['train'][0][:5])
    info_dict = json.load(open(data_path + 'sample_info_dict.json'))
    if 'dblp' in data_path:
        valid_paper = set([paper for paper in info_dict if int(info_dict[paper]['year']) <= train_time])
    elif 'pubmed' in data_path:
        valid_paper = set([paper for paper in info_dict if int(info_dict[paper]['pub_date']['year']) <= train_time])

    print(len(valid_paper))
    all_values = np.array(samples['train'][1])
    all_labels = np.ones(len(all_values))
    all_labels[all_values < 10] = 0
    all_labels[all_values >= 100] = 2

    all_paper_citation = {}
    all_paper_ref = {}
    for paper in samples['train'][0]:
        all_paper_citation[paper] = set([cite for cite in cite_dict.get(paper, []) if cite in valid_paper])
        all_paper_ref[paper] = set([ref for ref in ref_dict.get(paper, []) if ref in valid_paper])

    all_labels_dict = dict(zip(samples['train'][0], all_labels.tolist()))

    pub_time = {paper: [] for paper in samples['train'][0]}
    co_cite = {paper: [] for paper in samples['train'][0]}
    co_ref = {paper: [] for paper in samples['train'][0]}
    in_ref = {paper: [] for paper in samples['train'][0]}
    in_cite = {paper: [] for paper in samples['train'][0]}
    sample_len = len(samples['train'][0])
    for i in range(sample_len):
        paper = samples['train'][0][i]
        if 'dblp' in data_path:
            pub_time[paper] = int(info_dict[paper]['year'])
        elif 'pubmed' in data_path:
            pub_time[paper] = int(info_dict[paper]['pub_date']['year'])
        cur_cite = all_paper_citation[paper]
        cur_ref = all_paper_ref[paper]
        for j in range(i + 1, sample_len):
            comp_paper = samples['train'][0][j]
            comp_cite = all_paper_citation[comp_paper]
            comp_ref = all_paper_ref[comp_paper]
            if cur_cite & comp_cite:
                co_cite[paper].append(comp_paper)
                co_cite[comp_paper].append(paper)
            if cur_ref & comp_ref:
                co_ref[paper].append(comp_paper)
                co_ref[comp_paper].append(paper)
            if comp_paper in cur_ref:
                in_ref[paper].append(comp_paper)
                in_cite[comp_paper].append(paper)
            if comp_paper in cur_cite:
                in_cite[paper].append(comp_paper)
                in_ref[comp_paper].append(paper)

    df = pd.DataFrame(data={'pub_time': pub_time, 'label': all_labels_dict,
                            'co_cite': co_cite, 'co_ref': co_ref, 'in_cite': in_cite, 'in_ref': in_ref})
    # df['label'] = all_labels
    df.to_csv(data_path + 'hard_negative.csv')

    for i in range(sample_len):
        paper = samples['train'][0][i]
        cur_label = all_labels_dict[paper]
        for cur_dict in [co_cite, co_ref, in_cite, in_ref]:
            cur_dict[paper] = [comp_paper for comp_paper in cur_dict[paper] if all_labels_dict[comp_paper] != cur_label]
    df = pd.DataFrame(data={'pub_time': pub_time, 'label': all_labels_dict,
                            'co_cite': co_cite, 'co_ref': co_ref, 'in_cite': in_cite, 'in_ref': in_ref})
    # df['label'] = all_labels
    df.to_csv(data_path + 'hard_negative_labeled.csv')


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')

    parser.add_argument('--phase', default='test', help='the function name.')
    parser.add_argument('--data_source', default='h_pubmed', help='the data source.')
    parser.add_argument('--seed', default=123, help='the data seed.')
    parser.add_argument('--graph', default='random', help='the graph embed.')
    parser.add_argument('--structure', default='paper', help='the cascade structure.')

    args = parser.parse_args()
    print(args)
    data_source = args.data_source
    if args.data_source.startswith('h_'):
        data_source = args.data_source.replace('h_', '')
        default_config = get_configs(data_source, [])['default']
        train_time = default_config['time'][0]
        replace_dict = {'time': [train_time, train_time + 2, train_time + 4]}
    else:
        replace_dict = None
    configs = get_configs(data_source, ['H2CGL'], replace_dict)

    if args.phase == 'test':
        times = [2011, 2012, 2013]
        # configs['BDCTSGCN']['batch_graph'] = 'emb'
        configs['BDCTSGCN']['graph_name'] = 'graph_sample_feature_vector'
        dataProcessor = DataProcessor('h_pubmed', norm=True, time=times)
        graph_dict = dataProcessor.load_graphs('graph_sample_feature_vector')
        dataProcessor.get_selected_ids()
        dataloaders = dataProcessor.get_dataloader(32, graph_dict=graph_dict)
    elif args.phase == 'make_data':
        temp_config = configs['default']
        make_data(args.data_source, temp_config, seed=args.seed)
    elif args.phase == 'make_data_graph':
        temp_config = configs['default']
        if args.graph == 'vector':
            temp_config['tokenizer_type'] = 'glove'
            temp_config['tokenizer_path'] = './data/glove'
        elif args.graph == 'bert':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/scibert/'
        elif args.graph == 'bw':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/scibert/'
        elif args.graph == 'sbert':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/all-MiniLM-L12-v2/'
        elif args.graph == 'sbw':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/all-mpnet-base-v2/'
        else:
            temp_config['tokenizer_type'] = 'glove'
            temp_config['tokenizer_path'] = './data/glove'
        make_data(args.data_source, temp_config, seed=args.seed, graph=args.graph)
    elif args.phase == 'get_embs':
        temp_config = configs['default']
        if args.graph == 'bw':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/scibert/'
        elif args.graph == 'sbw':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/all-mpnet-base-v2/'
        elif args.graph == 'sbert':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/all-MiniLM-L12-v2/'
        dataProcessor = DataProcessor(args.data_source, norm=True)
        dataProcessor.get_embs(temp_config['tokenizer_type'], temp_config['tokenizer_path'],
                               mode=args.graph, split_abstract=temp_config['split_abstract'])
    elif args.phase == "get_cascade_graph":
        cascadeProcessor = CascadeDataProcessor(args.data_source, norm=True, max_len=10,
                                                max_time=configs['default']['time_length'])
        cascadeProcessor.get_cascade_graph(all_times=configs['default']['time'], structure=args.structure,
                                           time_length=configs['default']['time_length'])
    elif args.phase == 'show_graph_info':
        dataProcessor = DataProcessor(args.data_source, norm=True)
        dataProcessor.show_graph_info()
    elif args.phase == 'get_hn':
        dataProcessor = DataProcessor(args.data_source, norm=True, time=configs['default']['time'])
        show_hn(dataProcessor.data_cat_path, train_time=dataProcessor.time[0])

    # elif args.phase == 'get_node_embed':
    #     walks = list()
    #     read_walks_set('train', walks)
    #     learn_embeddings(walks, opts.dimensions)

    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
