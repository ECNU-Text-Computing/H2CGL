import torch
import dgl

from layers.GCN.DTGCN_layers import DCTSGCNLayer, SimpleDCTSGCNLayer
from our_models.CTSGCN import CTSGCN


class DCTSGCN(CTSGCN):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_index, word2vec=None, dropout=0.5, n_layers=3,
                 pred_type='paper', time_pooling='mean', hop=2, linear_before_gcn=False, encoder_type='GCN',
                 time_length=10, hidden_dim=300, out_dim=256, ntypes=None, etypes=None, edge_sequences=None,
                 graph_type=None, gcn_out='last', snapshot_weight=True, hn=None, hn_method=None, max_time_length=5,
                 **kwargs):
        super(DCTSGCN, self).__init__(vocab_size, embed_dim, num_classes, pad_index, word2vec, dropout, n_layers,
                                      pred_type, time_pooling, hop, linear_before_gcn, 'GCN',
                                      time_length, hidden_dim, out_dim, ntypes, etypes, graph_type, gcn_out, hn,
                                      hn_method, max_time_length,
                                      **kwargs)
        self.model_name = self.model_name.replace('CTSGCN', 'DCTSGCN').replace('_GCN', '_' + encoder_type) + \
                          '_{}'.format(edge_sequences)
        self.model_type = 'DCTSGCN'
        if snapshot_weight:
            self.model_name += '_weighted'
        if edge_sequences == 'meta':
            edge_sequences = [[('meta', [('author', 'writes', 'paper'),
                                         ('journal', 'publishes', 'paper'),
                                         ('time', 'shows', 'paper'),
                                         ('paper', 'is writen by', 'author'),
                                         ('paper', 'is published by', 'journal'),
                                         ('paper', 'is shown in', 'time')]),
                               ('inter_snapshots', [('paper', 'cites', 'paper'),
                                                    ('paper', 'is in', 'snapshot'),
                                                    ('paper', 'is cited by', 'paper'),
                                                    ('snapshot', 'has', 'paper')]),
                               ('intra_snapshots', [('snapshot', 't_cites', 'snapshot'),
                                                    ('snapshot', 'is t_cited by', 'snapshot')])]]
        elif edge_sequences == 'no_gcn':
            edge_sequences = [[('inter_snapshots', [('author', 'writes', 'paper'),
                                                    ('journal', 'publishes', 'paper'),
                                                    ('time', 'shows', 'paper'),
                                                    ('paper', 'cites', 'paper'),
                                                    ('paper', 'is in', 'snapshot'),
                                                    ('paper', 'is cited by', 'paper'),
                                                    ('paper', 'is writen by', 'author'),
                                                    ('paper', 'is published by', 'journal'),
                                                    ('paper', 'is shown in', 'time'),
                                                    ('snapshot', 'has', 'paper')])]]
        else:
            # basic
            edge_sequences = [[('inter_snapshots', [('author', 'writes', 'paper'),
                                                    ('journal', 'publishes', 'paper'),
                                                    ('time', 'shows', 'paper'),
                                                    ('paper', 'cites', 'paper'),
                                                    ('paper', 'is in', 'snapshot'),
                                                    ('paper', 'is cited by', 'paper'),
                                                    ('paper', 'is writen by', 'author'),
                                                    ('paper', 'is published by', 'journal'),
                                                    ('paper', 'is shown in', 'time'),
                                                    ('snapshot', 'has', 'paper')]),
                               ('intra_snapshots', [('snapshot', 't_cites', 'snapshot'),
                                                    ('snapshot', 'is t_cited by', 'snapshot')])]]
        etypes_set = set(etypes)
        for i in range(len(edge_sequences)):
            for j in range(len(edge_sequences[i])):
                # print('-'*30)
                cur_edges = edge_sequences[i][j][1].copy()
                for etype in cur_edges:
                    # print(edge_sequences[i][j])
                    # print(etype)
                    if etype[1] not in etypes_set:
                        edge_sequences[i][j][1].remove(etype)
                        print('removing', edge_sequences[i][j][0], etype)
        print(edge_sequences)
        combine_method = 'sum'
        # if len(edge_sequences) == 1:
        #     self.model_name += '_single'
        self.graph_encoder = None
        # self.graph_encoder = DCTSGCNLayer(in_feat=embed_dim, hidden_feat=hidden_dim, out_feat=out_dim, ntypes=ntypes,
        #                           edge_sequences=edge_sequences, k=n_layers, time_length=time_length).to(self.device)

        encoder_dict = {}
        if encoder_type == 'GIN':
            # ablation
            encoder_dict = {('paper', 'is in', 'snapshot'): {'cur_type': 'GAT'}}
        elif encoder_type == 'DGINP':
            # CGIN+RGAT
            encoder_dict = {('paper', 'is in', 'snapshot'): {'cur_type': 'RGAT',
                                                             'snapshot_types': 3,
                                                             'node_attr_types': ['is_cite', 'is_ref', 'is_target']
                                                             },
                            ('paper', 'cites', 'paper'): {'cur_type': 'DGINP'},
                            # ('paper', 'is cited by', 'paper'): {'cur_type': 'DGIN'}
                            }
            encoder_type = 'GIN'

        return_list = False if gcn_out == 'last' else True
        self.graph_encoder = SimpleDCTSGCNLayer(in_feat=self.embed_dim, hidden_feat=self.hidden_dim, out_feat=self.out_dim,
                                                ntypes=ntypes,
                                                edge_sequences=edge_sequences, k=n_layers, time_length=time_length,
                                                combine_method=combine_method, encoder_type=encoder_type,
                                                encoder_dict=encoder_dict, return_list=return_list,
                                                snapshot_weight=snapshot_weight, **kwargs).to(self.device)

