import logging
import time

import dgl
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from layers.GCN.CL_layers import *
from models.base_model import BaseModel
from utilis.scripts import custom_node_unbatch


class CLWrapper(BaseModel):
    def __init__(self, encoder: BaseModel, cl_type='simple', aug_type=None, tau=0.4, cl_weight=0.5, aug_rate=0.1, **kwargs):
        super(CLWrapper, self).__init__(0, 0, 1, 0, None, 0, None, **kwargs)
        self.encoder = encoder
        self.aux_criterion = encoder.aux_criterion
        self.aug_rate = aug_rate
        print('aug_rate:', self.aug_rate)
        if aug_type == 'md':
            self.aug_methods = (('attr_mask',), ('edge_drop',))
            self.aug_configs = ({
                                    'attr_mask': {
                                        'prob': self.aug_rate,
                                        'types': ['paper', 'author', 'journal', 'time']
                                    }
                                }, {
                                    'edge_drop': {
                                        'prob': self.aug_rate,
                                        'types': ["cites", "is cited by", "writes", "is writen by", "publishes",
                                                  "is published by",
                                                  "shows", "is shown in"]
                                    }
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
                                        'types': ["cites", "is cited by", "writes", "is writen by", "publishes",
                                                  "is published by",
                                                  "shows", "is shown in"]
                                    }

                                }, {
                                    'attr_mask': {
                                        'prob': self.aug_rate,
                                        'types': ['paper', 'author', 'journal', 'time']
                                    },
                                    'edge_drop': {
                                        'prob': self.aug_rate,
                                        'types': ["cites", "is cited by", "writes", "is writen by", "publishes",
                                                  "is published by",
                                                  "shows", "is shown in"]
                                    }
                                })
        elif aug_type == 'dd':
            self.aug_methods = (('attr_drop', 'edge_drop'), ('attr_drop', 'edge_drop'))
            self.aug_configs = ({
                                    'attr_drop': {
                                        'prob': self.aug_rate,
                                        'types': ['paper', 'author', 'journal', 'time']
                                    },
                                    'edge_drop': {
                                        'prob': self.aug_rate,
                                        'types': ["cites", "is cited by", "writes", "is writen by", "publishes",
                                                  "is published by",
                                                  "shows", "is shown in"]
                                    }
                                }, {
                                    'edge_drop': {
                                        'prob': self.aug_rate,
                                        'types': ["cites", "is cited by", "writes", "is writen by", "publishes",
                                                  "is published by",
                                                  "shows", "is shown in"]
                                    },
                                    'attr_drop': {
                                        'prob': self.aug_rate,
                                        'types': ['paper', 'author', 'journal', 'time']
                                    },
                                })
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
                                        'types': ["cites", "is cited by", "writes", "is writen by", "publishes",
                                                  "is published by",
                                                  "shows", "is shown in"]
                                    }
                                })
        print(self.aug_methods)
        self.model_name = self.encoder.model_name + '_CL_' + cl_type + '_' + aug_type
        if self.aug_rate != 0.1:
            self.model_name += '_p{}'.format(self.aug_rate)
        self.similarity = 'cos'
        self.cl_type = cl_type
        self.encoder.cl_type = self.cl_type
        self.tau = tau
        if self.tau != 0.4:
            self.model_name += '_t{}'.format(self.tau)
        self.cross_weight = 0.1
        self.cl_weight = cl_weight
        if self.cl_weight != 0.5:
            self.model_name += '_cw{}'.format(self.cl_weight)
        self.df_hn = None
        if cl_type in ['simple', 'cross_snapshot']:
            self.cl_ntypes = self.encoder.ntypes
        elif cl_type.startswith('focus'):
            self.cl_ntypes = [self.encoder.pred_type]
        elif cl_type.endswith('hard_negative'):
            self.cl_ntypes = self.encoder.ntypes
            print('wip')
        # self.cl_modules = nn.ModuleDict({ntype: CommonCLLayer(self.encoder.out_dim, self.encoder.out_dim, tau=self.tau)
        #                                      for ntype in self.cl_ntypes})
        self.proj_heads = nn.ModuleDict({ntype: ProjectionHead(self.encoder.out_dim, self.encoder.out_dim) for
                                         ntype in self.cl_ntypes})
        print('cur cl mode:', cl_type)
        print('weights:', self.cl_weight, self.cross_weight)

    def load_graphs(self, graphs):
        dealt_graphs = self.encoder.load_graphs(graphs)
        dealt_graphs = self.encoder.aug_graphs(dealt_graphs, self.aug_configs, self.aug_methods)
        df_hn = pd.read_csv('./data/{}/hard_negative.csv'.format(graphs['data_source']), index_col=0)
        paper_trans = graphs['node_trans']['paper']
        df_hn.index = [paper_trans[str(paper)] for paper in df_hn.index]
        for column in df_hn:
            df_hn[column] = df_hn[column].apply(lambda x: [paper_trans[str(paper)] for paper in eval(x)])
        self.df_hn = df_hn
        return dealt_graphs

    def graph_to_device(self, graphs):
        return self.encoder.graph_to_device(graphs)

    def deal_graphs(self, graphs, data_path=None, path=None, log_path=None):
        return self.encoder.deal_graphs(graphs, data_path, path, log_path)

    def get_aux_pred(self, pred):
        return self.encoder.get_aux_pred(pred)

    def get_new_gt(self, gt):
        return self.encoder.get_new_gt(gt)

    def get_aux_loss(self, pred, gt):
        return self.encoder.get_aux_loss(pred, gt)

    def get_cl_inputs(self, snapshots, require_grad=True):
        if require_grad:
            if type(snapshots) == list:
                for snapshot in snapshots:
                    for ntype in self.cl_ntypes:
                        snapshot.nodes[ntype].data['h'] = self.proj_heads[ntype](snapshot.nodes[ntype].data['h'])
            else:
                for ntype in self.cl_ntypes:
                    snapshots.nodes[ntype].data['h'] = self.proj_heads[ntype](snapshots.nodes[ntype].data['h'])
        else:
            if type(snapshots) == list:
                for snapshot in snapshots:
                    for ntype in self.cl_ntypes:
                        snapshot.nodes[ntype].data['h'] = self.proj_heads[ntype](snapshot.nodes[ntype].data['h']).detach()
            else:
                for ntype in self.cl_ntypes:
                    snapshots.nodes[ntype].data['h'] = self.proj_heads[ntype](snapshots.nodes[ntype].data['h']).detach()
        return snapshots

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
                    # print(self.cl_modules[ntype](z1, z2))
                    # if torch.isnan(temp_loss):
                    #     print(z1)
                    #     print(torch.isnan(z1).sum())
                    #     print(z2)
                    #     print(torch.isnan(z2).sum())
                # cl_loss_list.append()
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
                for t in range(self.encoder.time_length):
                    # start_time = time.time()
                    cur_z1s = []
                    cur_z2s = []
                    len_z1s = []
                    # len_z2s = []
                    t1_index = s1.nodes[ntype].data['snapshot_idx'] == t
                    t2_index = s2.nodes[ntype].data['snapshot_idx'] == t
                    for paper in ids:
                        s1_index = torch.where(t1_index & p1_index[paper.item()])
                        s2_index = torch.where(t2_index & p2_index[paper.item()])
                        z1 = s1.nodes[ntype].data['h'][s1_index]
                        z2 = s2.nodes[ntype].data['h'][s2_index]
                        cur_z1s.append(z1)
                        cur_z2s.append(z2)
                        cur_len = len(z1) if len(z1) > 0 else 1
                        len_z1s.append(cur_len)
                        # len_z2s.append(len(z2))
                    cur_z1s = pad_sequence(cur_z1s, batch_first=True)
                    cur_z2s = pad_sequence(cur_z2s, batch_first=True)
                    len_z1s = torch.tensor(len_z1s).to(cur_z1s.device)
                    # len_z2s = torch.tensor(len_z2s).to(cur_z2s.device)
                    # print('index_time: {:.3f}'.format(time.time() - start_time))
                    temp_loss = simple_cl(cur_z1s, cur_z2s, self.tau, valid_lens=len_z1s)
                    # print('loss_time: {:.3f}'.format(time.time() - start_time))
                    cur_loss = cur_loss + temp_loss
                # print(ntype, cur_loss / self.encoder.time_length)
                cl_loss = cl_loss + cur_loss / self.encoder.time_length
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

    def forward(self, content, lengths, masks, ids, graph, times, **kwargs):
        # cur_time = time.time()
        # g1_snapshots, g2_snapshots, cur_snapshot_adj = self.get_graphs(ids, graph)
        # g1_inputs = self.encoder.get_graphs(ids, graph, aug=(self.aug_methods[0], self.aug_configs[0]))
        # g2_inputs = self.encoder.get_graphs(ids, graph, aug=(self.aug_methods[1], self.aug_configs[1]))
        if self.cl_type.endswith('aug_hard_negative'):
            g1_inputs, g2_inputs = self.encoder.get_graphs(content, lengths, masks, ids, graph, times, phase='train',
                                                           df_hn=self.df_hn)
        else:
            g1_inputs, g2_inputs = self.encoder.get_graphs(content, lengths, masks, ids, graph, times, phase='train')
        # logging.info('aug done:{:.3f}s'.format(time.time() - cur_time))

        # cur_time = time.time()
        # g1_out, g1_snapshots = self.encode(g1_inputs)
        # g2_out, g2_snapshots = self.encode(g2_inputs)
        g1_out, g1_snapshots = self.encoder.encode(g1_inputs)
        g2_out, g2_snapshots = self.encoder.encode(g2_inputs)
        # logging.info('encode done:{:.3f}s'.format(time.time() - cur_time))

        # cur_time = time.time()
        g1_snapshots, g2_snapshots = self.get_cl_inputs(g1_snapshots), self.get_cl_inputs(g2_snapshots)
        self.other_loss = self.cl_weight * self.cl(g1_snapshots, g2_snapshots, ids)
        # logging.info('cl done:{:.3f}s'.format(time.time() - cur_time))

        time_out = (g1_out + g2_out) / 2
        output = self.encoder.decode(time_out)
        # print(time_out.shape)
        # logging.info('=' * 20 + 'finish' + '=' * 20)
        return output

    def predict(self, content, lengths, masks, ids, graph, times, **kwargs):
        # print(final_out.shape)
        inputs = self.encoder.get_graphs(content, lengths, masks, ids, graph, times)
        # print(inputs)
        time_out, _ = self.encoder.encode(inputs)
        output = self.encoder.decode(time_out)
        print(output[0].shape)
        return output

    def show(self, content, lengths, masks, ids, graph, times, **kwargs):
        output = self.encoder.show(content, lengths, masks, ids, graph, times, **kwargs)
        return output


def get_unbatched_edge_graphs(batched_graph, ntype, etype):
    subgraph = dgl.edge_type_subgraph(batched_graph, etypes=[etype])
    subgraph.set_batch_num_nodes(batched_graph.batch_num_nodes(ntype))
    # print(batched_graph.batch_num_edges('cites'))
    subgraph.set_batch_num_edges(batched_graph.batch_num_edges(etype))
    # print(subgraph)
    # print(dgl.unbatch(subgraph))
    return dgl.unbatch(subgraph)
