import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from layers.GCN.CL_layers import simple_cl, ProjectionHead
from models.base_model import BaseModel
from our_models.CL import CLWrapper

class HCLWrapper(CLWrapper):
    def __init__(self, encoder: BaseModel, cl_type='simple', aug_type=None, tau=0.4, cl_weight=0.5, aug_rate=0.1, **kwargs):
        super(HCLWrapper, self).__init__(encoder, cl_type, 'normal', tau, cl_weight, aug_rate, **kwargs)
        self.model_name = self.model_name.replace('_CL_', '_HCL_').replace('_normal', '_' + aug_type)
        if self.cl_type in ['simple', 'cross_snapshot']:
            print('not support contrast for each type')
            raise Exception
        self.cl_ntypes = ['snapshot']
        self.proj_heads = nn.ModuleDict({ntype: ProjectionHead(self.encoder.out_dim, self.encoder.out_dim) for
                                         ntype in self.cl_ntypes})
        print('cur cl mode:', cl_type)
        # print('weights:', self.cl_weight, self.cross_weight)
        self.aug_type = aug_type

        if aug_type == 'dd':
            self.aug_methods = (('attr_drop', 'edge_drop'), ('attr_drop', 'edge_drop'))
            self.aug_configs = ({
                                    'attr_drop': {
                                        'prob': self.aug_rate,
                                        'types': ['paper', 'author', 'journal', 'time']
                                    },
                                    'edge_drop': {
                                        'prob': self.aug_rate,
                                        'types': [('paper', 'cites', 'paper'),
                                                  ('paper', 'is cited by', 'paper'),
                                                  ('author', 'writes', 'paper'),
                                                  ('paper', 'is writen by', 'author'),
                                                  ('journal', 'publishes', 'paper'),
                                                  ('paper', 'is published by', 'journal'),
                                                  ('time', 'shows', 'paper'),
                                                  ('paper', 'is shown in', 'time')]
                                    }
                                }, {
                                    'edge_drop': {
                                        'prob': self.aug_rate,
                                        'types': [('paper', 'cites', 'paper'),
                                                  ('paper', 'is cited by', 'paper'),
                                                  ('author', 'writes', 'paper'),
                                                  ('paper', 'is writen by', 'author'),
                                                  ('journal', 'publishes', 'paper'),
                                                  ('paper', 'is published by', 'journal'),
                                                  ('time', 'shows', 'paper'),
                                                  ('paper', 'is shown in', 'time')]
                                    },
                                    'attr_drop': {
                                        'prob': self.aug_rate,
                                        'types': ['paper', 'author', 'journal', 'time']
                                    },
                                })
        elif aug_type == 'dmd':
            self.aug_methods = (('attr_mask', 'edge_drop'), ('attr_mask', 'edge_drop'))
            self.aug_configs = ({
                                    'attr_mask': {
                                        'prob': self.aug_rate,
                                        'types': ['paper', 'author', 'journal', 'time']
                                    },
                                    'edge_drop': {
                                        'prob': self.aug_rate,
                                        'types': [('paper', 'cites', 'paper'),
                                                  ('paper', 'is cited by', 'paper'),
                                                  ('author', 'writes', 'paper'),
                                                  ('paper', 'is writen by', 'author'),
                                                  ('journal', 'publishes', 'paper'),
                                                  ('paper', 'is published by', 'journal'),
                                                  ('time', 'shows', 'paper'),
                                                  ('paper', 'is shown in', 'time')]
                                    }
                                }, {
                                    'edge_drop': {
                                        'prob': self.aug_rate,
                                        'types': [('paper', 'cites', 'paper'),
                                                  ('paper', 'is cited by', 'paper'),
                                                  ('author', 'writes', 'paper'),
                                                  ('paper', 'is writen by', 'author'),
                                                  ('journal', 'publishes', 'paper'),
                                                  ('paper', 'is published by', 'journal'),
                                                  ('time', 'shows', 'paper'),
                                                  ('paper', 'is shown in', 'time')]
                                    },
                                    'attr_mask': {
                                        'prob': self.aug_rate,
                                        'types': ['paper', 'author', 'journal', 'time']
                                    },
                                })
        elif aug_type == 'cg':
            self.aug_methods = (('node_drop',), ('node_drop',))
            self.aug_configs = ({
                'node_drop': {
                    'prob': self.aug_rate,
                    'types': []
                }}, {
                'node_drop': {
                    'prob': self.aug_rate,
                    'types': []
                }})
        else:
            aug_type = 'normal'
            self.aug_methods = (('attr_drop',), ('edge_drop',))
            self.aug_configs = ({
                                    'attr_drop': {
                                        'prob': self.aug_rate,
                                        'types': ['paper', 'author', 'journal', 'time']
                                    }
                                }, {
                                    'edge_drop': {
                                        'prob': self.aug_rate,
                                        'types': [('paper', 'cites', 'paper'),
                                                  ('paper', 'is cited by', 'paper'),
                                                  ('author', 'writes', 'paper'),
                                                  ('paper', 'is writen by', 'author'),
                                                  ('journal', 'publishes', 'paper'),
                                                  ('paper', 'is published by', 'journal'),
                                                  ('time', 'shows', 'paper'),
                                                  ('paper', 'is shown in', 'time')]
                                    }
                                })
        print(self.aug_methods)

    def load_graphs(self, graphs):
        dealt_graphs = self.encoder.load_graphs(graphs)
        if self.aug_type != 'cg':
            dealt_graphs = self.encoder.aug_graphs_plus(dealt_graphs, self.aug_configs, self.aug_methods)
        paper_trans = graphs['node_trans']['paper']
        if 'label' in self.cl_type:
            df_hn = pd.read_csv('./data/{}/hard_negative_labeled.csv'.format(graphs['data_source']), index_col=0)
            df_hn.index = [paper_trans[str(paper)] for paper in df_hn.index]
            self.encoder.re_group = {label: list(df_hn[df_hn['label'] != label].index) for label in [0, 1, 2]}
        else:
            df_hn = pd.read_csv('./data/{}/hard_negative.csv'.format(graphs['data_source']), index_col=0)
            df_hn.index = [paper_trans[str(paper)] for paper in df_hn.index]
        for column in df_hn:
            if column in ['pub_time', 'label']:
                df_hn[column] = df_hn[column].astype(int)
            else:
                df_hn[column] = df_hn[column].apply(lambda x: [paper_trans[str(paper)] for paper in eval(x)])
                # df_hn[column] = df_hn[column].astype('object')
        self.df_hn = df_hn
        return dealt_graphs

    def cl(self, s1, s2, ids):
        # cl_loss_list = []
        cl_loss = 0
        z1_snapshots, z2_snapshots = [], []
        z1_ids, z2_ids = [], []
        if self.cl_type in ['simple', 'focus', 'cross_snapshot', 'focus_cross_snapshot']:
            for ntype in self.cl_ntypes:
                cur_loss = 0
                for t in range(self.encoder.time_length):
                    if 'HTSGCN' in self.encoder.model_name:
                        if ntype == 'snapshot':
                            s1_index = torch.where(s1[t].nodes[ntype].data['mask'] > 0)
                            s2_index = torch.where(s2[t].nodes[ntype].data['mask'] > 0)
                            z1 = s1[t].nodes[ntype].data['h'][s1_index]
                            z2 = s2[t].nodes[ntype].data['h'][s2_index]
                        else:
                            z1 = s1[t].nodes[ntype].data['h']
                            z2 = s2[t].nodes[ntype].data['h']
                    elif 'CTSGCN' in self.encoder.model_name:
                        s1_index = torch.where(s1.nodes[ntype].data['snapshot_idx'] == t)
                        s2_index = torch.where(s2.nodes[ntype].data['snapshot_idx'] == t)
                        # print(s1_index)
                        z1 = s1.nodes[ntype].data['h'][s1_index]
                        z2 = s2.nodes[ntype].data['h'][s2_index]
                        # print(z1.shape)
                    if (self.cl_type.endswith('cross_snapshot')) and (ntype == 'snapshot'):
                        z1_snapshots.append(z1)
                        z2_snapshots.append(z2)
                        z1_ids.append(s1.nodes[ntype].data['target'][s1_index])
                        z2_ids.append(s2.nodes[ntype].data['target'][s2_index])
                    # cur_list.append(self.cl_modules[ntype](z1, z2))
                    # temp_loss = self.cl_modules[ntype](z1, z2)
                    temp_loss = simple_cl(z1, z2, self.tau)
                    cur_loss = cur_loss + temp_loss

                cl_loss = cl_loss + cur_loss / self.encoder.time_length
            cl_loss = cl_loss / len(self.cl_ntypes)
        elif self.cl_type.endswith('hard_negative'):
            for ntype in self.cl_ntypes:
                cur_loss = 0
                p1_index = {}
                p2_index = {}
                for paper in ids:
                    p1_index[paper.item()] = s1.nodes[ntype].data['target'] == paper.item()
                    p2_index[paper.item()] = s2.nodes[ntype].data['target'] == paper.item()
                # print(p1_index.keys())
                time_length = 0
                for t in range(self.encoder.time_length):
                    # start_time = time.time()
                    cur_z1s = []
                    cur_z2s = []
                    len_z1s = []
                    # len_z2s = []
                    t1_index = s1.nodes[ntype].data['snapshot_idx'] == t
                    t2_index = s2.nodes[ntype].data['snapshot_idx'] == t
                    for paper in ids:
                        # s1_index = torch.where(t1_index & p1_index[paper.item()])
                        # s2_index = torch.where(t2_index & p2_index[paper.item()])
                        # z1 = s1.nodes[ntype].data['h'][s1_index]
                        # z2 = s2.nodes[ntype].data['h'][s2_index]
                        # cur_z1s.append(z1)
                        # cur_z2s.append(z2)
                        # cur_len = len(z1) if len(z1) > 0 else 1
                        # len_z1s.append(cur_len)
                        # print(s1_index[0].shape[0], cur_len)
                        s1_index = torch.where(t1_index & p1_index[paper.item()])
                        s2_index = torch.where(t2_index & p2_index[paper.item()])
                        if s1_index[0].shape[0] >= 2:
                        # if s1_index[0].shape[0] >= self.encoder.hn:
                            z1 = s1.nodes[ntype].data['h'][s1_index]
                            z2 = s2.nodes[ntype].data['h'][s2_index]
                            cur_z1s.append(z1)
                            cur_z2s.append(z2)
                            cur_len = len(z1)
                            len_z1s.append(cur_len)
                            # print(s1_index[0].shape[0], cur_len)
                        # len_z2s.append(len(z2))
                        # if s1_index[0].shape[0] > 2:
                        #     print('sidx:', (s1.nodes[ntype].data['snapshot_idx'] == 9).sum())
                        #     print(ntype)
                        #     print(torch.where(p1_index[paper.item()]))
                        #     print(p1_index[paper.item()].sum())
                        #     print(s1_index)
                    if len(cur_z1s) > 0:
                        cur_z1s = pad_sequence(cur_z1s, batch_first=True)
                        cur_z2s = pad_sequence(cur_z2s, batch_first=True)
                        len_z1s = torch.tensor(len_z1s).to(cur_z1s.device)
                        # len_z2s = torch.tensor(len_z2s).to(cur_z2s.device)
                        # print('index_time: {:.3f}'.format(time.time() - start_time))
                        # print(cur_z1s.shape)
                        temp_loss = simple_cl(cur_z1s, cur_z2s, self.tau, valid_lens=len_z1s)
                        time_length = time_length + 1
                    else:
                        temp_loss = 0
                    # print(temp_loss)
                    # print('loss_time: {:.3f}'.format(time.time() - start_time))
                    cur_loss = cur_loss + temp_loss
                # print(ntype, cur_loss / self.encoder.time_length)
                cl_loss = cl_loss + cur_loss / time_length
            cl_loss = cl_loss / len(self.cl_ntypes)

        if self.cl_type.endswith('cross_snapshot'):
            snapshot_loss = 0
            z1_dict = {paper.item(): [] for paper in ids}
            z2_dict = {paper.item(): [] for paper in ids}
            # print(z1_dict)
            for i in range(len(z1_snapshots)):
                cur_z1 = z1_snapshots[i]
                cur_z2 = z2_snapshots[i]
                cur_z1_ids = z1_ids[i]
                cur_z2_ids = z2_ids[i]
                for j in range(len(cur_z1_ids)):
                    z1_dict[cur_z1_ids[j].item()].append(cur_z1[j])
                    z2_dict[cur_z2_ids[j].item()].append(cur_z2[j])
            for paper in ids:
                # print(torch.stack(z1_dict[paper.item()], dim=0).shape)
                snapshot_loss = snapshot_loss + simple_cl(torch.stack(z1_dict[paper.item()], dim=0),
                                                          torch.stack(z2_dict[paper.item()], dim=0),
                                                          self.tau)

            cl_loss = cl_loss + self.cross_weight * snapshot_loss
        # print(cl_loss)
        return cl_loss