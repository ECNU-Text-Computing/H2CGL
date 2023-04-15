import argparse
import datetime
from collections import defaultdict

import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.hetero import get_aggregate_fn

from layers.GCN.RGCN import StochasticKLayerRGCN, CustomGCN


class DCTSGCNLayer(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat=None, ntypes=None, edge_sequences=None, k=3, time_length=10):
        super().__init__()
        # T * k layers
        self.k = k
        self.time_length = time_length
        self.ntypes = ntypes
        if out_feat is None:
            out_feat = hidden_feat
        if edge_sequences:
            forward_sequences, backward_sequences = edge_sequences
            self.concat_fc = nn.ModuleList(
                [dglnn.HeteroLinear({ntype: hidden_feat * 2 for ntype in ntypes}, hidden_feat)])
            # print('fc', self.concat_fc[0].linears['author'].weight.device)
            self.forward_layers = nn.ModuleList([DirectedCTSGCNLayer(ntypes, time_length, forward_sequences,
                                                                     in_feat, hidden_feat)])
            self.backward_layers = nn.ModuleList([DirectedCTSGCNLayer(ntypes, time_length, backward_sequences,
                                                                      in_feat, hidden_feat)])
            for i in range(k - 2):
                self.forward_layers.append(DirectedCTSGCNLayer(ntypes, time_length, forward_sequences,
                                                               hidden_feat, hidden_feat))
                self.backward_layers.append(DirectedCTSGCNLayer(ntypes, time_length, backward_sequences,
                                                                hidden_feat, hidden_feat))
                self.concat_fc.append(dglnn.HeteroLinear({ntype: hidden_feat * 2 for ntype in ntypes}, hidden_feat))
            self.forward_layers.append(DirectedCTSGCNLayer(ntypes, time_length, forward_sequences,
                                                           hidden_feat, out_feat))
            self.backward_layers.append(DirectedCTSGCNLayer(ntypes, time_length, backward_sequences,
                                                            hidden_feat, out_feat))
            self.concat_fc.append(dglnn.HeteroLinear({ntype: out_feat * 2 for ntype in ntypes}, out_feat))

        # self.concat_fc = self.concat_fc.to(device)
        # print('fc', self.concat_fc[0].linears['author'].weight.device)
        # self.forward_layers = self.forward_layers.to(device)
        # self.backward_layers = self.backward_layers.to(device)

    def forward(self, snapshots, input_feats):
        hidden_embs = input_feats
        # hidden_embs = snapshots.srcdata['h']
        for i in range(self.k):
            # print('-'*30, i, '-'*30)
            # print([emb.shape for emb in hidden_embs[0].values()])
            forward_embs = self.forward_layers[i](snapshots, hidden_embs)
            # print('=' * 61)
            # print([emb.shape for emb in hidden_embs[0].values()])
            backward_embs = self.backward_layers[i](snapshots, hidden_embs)
            # out_embs = [
            #     {ntype: torch.cat((forward_embs[t][ntype], backward_embs[t][ntype]), dim=-1) for ntype in self.ntypes}
            #     for t in range(self.time_length)]
            out_embs = {ntype: torch.cat((forward_embs[ntype], backward_embs[ntype]), dim=-1) for ntype in self.ntypes}
            # print('here', out_embs[0])
            # hidden_embs = [self.concat_fc[i](out_embs[t]) for t in range(self.time_length)]
            hidden_embs = self.concat_fc[i](out_embs)

        return hidden_embs

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        print('whole graph_encoder to device')
        if self.concat_fc:
            self.concat_fc = self.concat_fc.to(*args, **kwargs)
        for i in range(len(self.forward_layers)):
            self.forward_layers[i] = self.forward_layers[i].to(*args, **kwargs)
        # print(self.forward_layers)
        for i in range(len(self.backward_layers)):
            self.backward_layers[i] = self.backward_layers[i].to(*args, **kwargs)
        return self


class DirectedCTSGCNLayer(nn.Module):
    def __init__(self, ntypes, time_length, edge_sequences, in_feat, hidden_feat, out_feat=None, agg_method='sum',
                 activation=torch.relu):
        super(DirectedCTSGCNLayer, self).__init__()
        if not out_feat:
            out_feat = hidden_feat
        self.time_length = time_length
        # meta, paper, to_snapshot, snapshot
        self.sequence = [edges_type for edges_type, _ in edge_sequences]
        self.edges_type = {edges_type: etypes for edges_type, etypes in edge_sequences}
        self.agg_method = get_aggregate_fn(agg_method)
        self.snapshot_weight = 'none'
        self.convs = nn.ModuleDict({})
        self.activation = activation
        self.skip_connect_fc = nn.ModuleDict({edges_type: nn.ModuleDict({}) for edges_type in self.edges_type})

        trans_nodes = {}
        input_dim = {ntype: in_feat for ntype in ntypes}
        # if self.sequence[0] != 'snapshot':
        #     ntypes = [stype for stype, _, _ in edge_sequences[0][-1]]
        #     trans_nodes = {ntype: in_feat for ntype in ntypes}
        # for ntype in ntypes:
        #     if ntype not in trans_nodes:
        #         trans_nodes[ntype] = out_feat
        for i in range(len(edge_sequences)):
            edges_type, etypes = edge_sequences[i]
            if edges_type != 'intra_snapshots':
                # if i == 0:
                #     self.convs[edges_type] = nn.ModuleList([dglnn.HeteroGraphConv(
                #         {etype: dglnn.GraphConv(in_feat, hidden_feat, norm='right')
                #                                             for etype in etypes}) for t in range(time_length)])
                # elif i == len(edge_sequences) - 1:
                #     self.convs[edges_type] = nn.ModuleList([dglnn.HeteroGraphConv(
                #         {etype: dglnn.GraphConv(hidden_feat, out_feat, norm='right')
                #          for etype in etypes})  for t in range(time_length)])
                # else:
                #     self.convs[edges_type] = nn.ModuleList([dglnn.HeteroGraphConv(
                #         {etype: dglnn.GraphConv(hidden_feat, hidden_feat, norm='right')
                #          for etype in etypes}) for t in range(time_length)])
                if i == 0:
                    self.convs[edges_type] = nn.ModuleDict({str(etype):
                                                                dglnn.GraphConv(in_feat, hidden_feat, norm='right',
                                                                                allow_zero_in_degree=True)
                                                            for etype in etypes})
                    cur_out = hidden_feat
                    # print(list(self.convs[edges_type].values())[0][0].weight.device)
                elif i == len(edge_sequences) - 1:
                    self.convs[edges_type] = nn.ModuleDict({str(etype):
                                                                dglnn.GraphConv(hidden_feat, out_feat, norm='right',
                                                                                allow_zero_in_degree=True)
                                                            for etype in etypes})
                    cur_out = out_feat
                else:
                    self.convs[edges_type] = nn.ModuleDict({
                        str(etype):
                            dglnn.GraphConv(hidden_feat, hidden_feat, norm='right', allow_zero_in_degree=True)
                        for etype in etypes})
                    cur_out = hidden_feat
                for etype in etypes:
                    trans_nodes[etype[-1]] = cur_out
                    self.skip_connect_fc[edges_type][etype[-1]] = nn.Linear(input_dim[etype[-1]], cur_out)
                for etype in etypes:
                    input_dim[etype[-1]] = cur_out
            else:
                if self.snapshot_weight:
                    snapshot_norm = 'none'
                else:
                    snapshot_norm = 'right'
                if i == 0:
                    self.convs[edges_type] = dglnn.GraphConv(in_feat, hidden_feat, norm=snapshot_norm)
                    cur_out = hidden_feat
                else:
                    self.convs[edges_type] = dglnn.GraphConv(hidden_feat, out_feat, norm=snapshot_norm)
                    cur_out = out_feat
                trans_nodes['snapshot'] = cur_out
                input_dim['snapshot'] = cur_out
                # self.skip_connect_fc[edges_type]['snapshot'] = nn.Linear(cur_in, cur_out)
        for ntype in ntypes:
            if ntype not in trans_nodes:
                trans_nodes[ntype] = in_feat
        # self.trans_fc = None
        self.trans_fc = dglnn.HeteroLinear(trans_nodes, out_feat)

    # def to(self, *args, **kwargs):
    #     self = super().to(*args, **kwargs)
    #     # for edges_type in self.convs:
    #     #     print(edges_type + ' to device')
    #     #     self.convs[edges_type] = self.convs[edges_type].to(*args, **kwargs)
    #     # self.convs = self.convs.to(*args, **kwargs)
    #     return self

    def normal_forward(self, snapshots, input_embs, edges_type):
        # conv_embs = []
        outputs = defaultdict(list)
        all_embs = input_embs
        for etype in self.edges_type[edges_type]:
            # print('etype', etype)
            stype, _, dtype = etype
            subgraph = snapshots[tuple(etype)]
            # subgraph = dgl.add_self_loop(subgraph)
            # print(etype, subgraph.in_degrees())
            srcdata = all_embs[stype]
            dstdata = all_embs[dtype]
            outputs[dtype].append(self.convs[edges_type][str(etype)](subgraph, (srcdata, dstdata)))
        for dtype in outputs:
            if len(outputs[dtype]) > 0:
                all_embs[dtype] = self.skip_connect_fc[edges_type][dtype](all_embs[dtype]) \
                                  + self.agg_method(outputs[dtype], dtype)
        # conv_embs.append(all_embs)

        return all_embs

    def snapshot_forward(self, input_embs, snapshot_graphs):
        ndata = input_embs['snapshot']
        subgraph = snapshot_graphs[tuple(self.edges_type['intra_snapshots'])]
        if self.snapshot_weight:
            norm = dglnn.EdgeWeightNorm(norm=self.snapshot_weight)
            norm_edge_weight = norm(subgraph, subgraph.edata['w'])
            # print(ndata.shape)
            # print(self.convs['intra_snapshots'].weight.shape)
            # print(ndata)
            # print(self.edges_type)
            conv_emb = self.convs['intra_snapshots'](subgraph, (ndata, ndata), edge_weight=norm_edge_weight)
        else:
            conv_emb = self.convs['intra_snapshots'](subgraph, (ndata, ndata))
        # dstdata = torch.split(conv_emb, batch_size, dim=0)  # T[B, d]
        # for t in range(self.time_length):
        #     input_embs[t]['snapshot'] = dstdata[t]
        input_embs['snapshot'] = conv_emb
        return input_embs

    def forward(self, snapshots, input_embs):
        # print('-'*59)
        # hidden_embs = [snapshot.srcdata['h'] for snapshot in snapshots]
        # copy for dual-direction
        input_embs = input_embs.copy()
        hidden_embs = input_embs
        # print(input_embs[0]['snapshot'])
        for edges_type in self.sequence:
            if edges_type != 'intra_snapshots':
                hidden_embs = self.normal_forward(snapshots, hidden_embs, edges_type)
            else:
                hidden_embs = self.snapshot_forward(hidden_embs, snapshots)
        if self.activation:
            for ntype in hidden_embs.keys():
                hidden_embs[ntype] = self.activation(hidden_embs[ntype])
        # 共用一个MLP
        hidden_embs = self.trans_fc(hidden_embs)
        # T[nodes.data]
        return hidden_embs


class SimpleDCTSGCNLayer(DCTSGCNLayer):
    def __init__(self, in_feat, hidden_feat, out_feat=None, ntypes=None, edge_sequences=None, k=3, time_length=10,
                 combine_method='concat', encoder_type='GCN', encoder_dict=None, return_list=False, snapshot_weight=None,
                 **kwargs):
        super().__init__(in_feat, hidden_feat, out_feat, ntypes, None, k, time_length)
        print(encoder_type, encoder_dict)
        # T * k layers
        self.k = k
        self.time_length = time_length
        self.ntypes = ntypes
        if out_feat is None:
            out_feat = hidden_feat
        self.fc_in = dglnn.HeteroLinear({ntype: in_feat for ntype in ntypes}, hidden_feat)
        # self.fc_out = dglnn.HeteroLinear({ntype: hidden_feat for ntype in ntypes}, out_feat)
        self.fc_out = nn.ModuleList([dglnn.HeteroLinear({ntype: hidden_feat for ntype in ntypes}, out_feat)
                                     for i in range(k)])
        self.return_list = return_list

        print(combine_method)
        self.concat_fc = None
        if combine_method == 'concat':
            self.concat_fc = nn.ModuleList(
                [dglnn.HeteroLinear({ntype: hidden_feat * 2 for ntype in ntypes}, hidden_feat)
                 for i in range(k)])
        # print('fc', self.concat_fc[0].linears['author'].weight.device)
        if len(edge_sequences) > 1:
            self.dual_view = True
            forward_sequences, backward_sequences = edge_sequences
            self.forward_layers = nn.ModuleList([SimpleDirectedCTSGCNLayer(time_length, forward_sequences, hidden_feat,
                                                                           encoder_type=encoder_type,
                                                                           encoder_dict=encoder_dict,
                                                                           snapshot_weight=snapshot_weight, **kwargs)
                                                 for i in range(k)])
            self.backward_layers = nn.ModuleList([SimpleDirectedCTSGCNLayer(time_length, forward_sequences, hidden_feat,
                                                                            encoder_type=encoder_type,
                                                                            encoder_dict=encoder_dict,
                                                                            snapshot_weight=snapshot_weight, **kwargs)
                                                  for i in range(k)])
        else:
            self.dual_view = False
            single_sequences = edge_sequences[0]
            self.forward_layers = []
            self.backward_layers = []
            self.single_layers = nn.ModuleList([SimpleDirectedCTSGCNLayer(time_length, single_sequences, hidden_feat,
                                                                          encoder_type=encoder_type,
                                                                          encoder_dict=encoder_dict,
                                                                          snapshot_weight=snapshot_weight, **kwargs)
                                                for i in range(k)])
        print('dual view:', self.dual_view)

    def forward(self, snapshots, hidden_embs, get_attention=False):
        # hidden_embs = snapshots.srcdata['h']
        output_list = []
        attn_list = []
        hidden_embs = self.fc_in(hidden_embs)
        for i in range(self.k):
            if self.dual_view:
                forward_embs = self.forward_layers[i](snapshots, hidden_embs)
                backward_embs = self.backward_layers[i](snapshots, hidden_embs)
                if self.concat_fc:
                    out_embs = {ntype: torch.cat((forward_embs[ntype], backward_embs[ntype]), dim=-1) for ntype in
                                self.ntypes}
                    hidden_embs = self.concat_fc[i](out_embs)
                else:
                    hidden_embs = {ntype: forward_embs[ntype] + backward_embs[ntype] for ntype in
                                   self.ntypes}
            else:
                if get_attention:
                    hidden_embs, attn = self.single_layers[i](snapshots, hidden_embs, get_attention=True)
                    attn_list.append(attn)
                else:
                    hidden_embs = self.single_layers[i](snapshots, hidden_embs)
            if self.return_list:
                output_list.append(self.fc_out[i](hidden_embs))
        hidden_embs = self.fc_out[-1](hidden_embs)
        # for key in hidden_embs:
        #     print(hidden_embs[key].shape)
        if get_attention:
            return hidden_embs, output_list, attn_list
        else:
            return hidden_embs, output_list


class SimpleDirectedCTSGCNLayer(nn.Module):
    def __init__(self, time_length, edge_sequences, hidden_feat, agg_method='sum', encoder_type='GCN',
                 encoder_dict=None, snapshot_weight=None,
                 activation=nn.LeakyReLU(), **kwargs):
        # simplified version
        super(SimpleDirectedCTSGCNLayer, self).__init__()
        self.time_length = time_length
        # meta, paper, to_snapshot, snapshot
        print(len(edge_sequences))
        self.sequence = [edges_type for edges_type, _ in edge_sequences]
        self.edges_type = {edges_type: etypes for edges_type, etypes in edge_sequences}
        self.agg_method = get_aggregate_fn(agg_method)
        self.snapshot_weight = snapshot_weight
        # self.snapshot_weight = None
        print('snapshot_weight', self.snapshot_weight)
        self.convs = nn.ModuleDict({})
        self.activation = activation

        for i in range(len(edge_sequences)):
            edges_type, etypes = edge_sequences[i]
            if edges_type != 'intra_snapshots':
                # self.convs[edges_type] = nn.ModuleDict({str(etype):
                #                                             dglnn.GraphConv(hidden_feat, hidden_feat, norm='right',
                #                                                             allow_zero_in_degree=True)
                #                                         for etype in etypes})
                # self.convs[edges_type] = nn.ModuleDict({str(etype):
                #                                             CustomGCN(encoder_type, hidden_feat, hidden_feat, layer=i+1,
                #                                                       activation=activation, **kwargs)
                #                                         for etype in etypes})
                self.convs[edges_type] = nn.ModuleDict({})
                for etype in etypes:
                    if etype not in encoder_dict:
                        cur_encoder_type = encoder_type
                        encoder_args = {}
                    else:
                        cur_encoder_type = encoder_dict[etype]['cur_type']
                        encoder_args = encoder_dict[etype]
                    self.convs[edges_type][str(etype)] = CustomGCN(cur_encoder_type, hidden_feat, hidden_feat,
                                                                   layer=i + 1, activation=activation, **encoder_args)

            else:
                if self.snapshot_weight:
                    self.snapshot_weight = 'none'
                else:
                    self.snapshot_weight = 'right'
                self.convs[edges_type] = nn.ModuleDict({})
                for etype in etypes:
                    if etype not in encoder_dict:
                        cur_encoder_type = encoder_type
                        encoder_args = {}
                    else:
                        cur_encoder_type = encoder_dict[etype]['cur_type']
                        encoder_args = encoder_dict[etype]
                        print(encoder_args)
                    self.convs[edges_type][str(etype)] = CustomGCN(cur_encoder_type, hidden_feat, hidden_feat,
                                                                   layer=i + 1, activation=activation,
                                                                   **encoder_args)

    def normal_forward(self, snapshots, input_embs, edges_type):
        # conv_embs = []
        outputs = defaultdict(list)
        all_embs = input_embs
        for etype in self.edges_type[edges_type]:
            # print('etype', etype)
            stype, _, dtype = etype
            subgraph = snapshots[tuple(etype)]
            # subgraph = dgl.add_self_loop(subgraph)
            # print(etype, subgraph.in_degrees())
            srcdata = all_embs[stype]
            dstdata = all_embs[dtype]
            outputs[dtype].append(self.convs[edges_type][str(etype)](subgraph, (srcdata, dstdata)))
        for dtype in outputs:
            if len(outputs[dtype]) > 0:
                all_embs[dtype] = self.agg_method(outputs[dtype], dtype)
        # conv_embs.append(all_embs)

        return all_embs

    def snapshot_forward(self, input_embs, snapshot_graphs):
        outputs = defaultdict(list)
        all_embs = input_embs
        for etype in self.edges_type['intra_snapshots']:
            # print('etype', etype)
            stype, _, dtype = etype
            subgraph = snapshot_graphs[tuple(etype)]
            # subgraph = dgl.add_self_loop(subgraph)
            # print(etype, subgraph.in_degrees())
            srcdata = all_embs[stype]
            dstdata = all_embs[dtype]
            norm_edge_weight = None
            if self.snapshot_weight:
                norm = dglnn.EdgeWeightNorm(norm=self.snapshot_weight)
                norm_edge_weight = norm(subgraph, subgraph.edata['w'])
            outputs[dtype].append(self.convs['intra_snapshots'][str(etype)](subgraph, (srcdata, dstdata),
                                                                            edge_weight=norm_edge_weight))
        for dtype in outputs:
            if len(outputs[dtype]) > 0:
                all_embs[dtype] = self.agg_method(outputs[dtype], dtype)
        return all_embs

    def get_attn(self, snapshots, input_embs, edges_type, etype):
        # conv_embs = []
        outputs = defaultdict(list)
        all_embs = input_embs
        # print('etype', etype)
        # etype = ('paper', 'is in', 'snapshot')
        stype, _, dtype = etype
        subgraph = snapshots[tuple(etype)]
        # subgraph = dgl.add_self_loop(subgraph)
        # print(etype, subgraph.in_degrees())
        srcdata = all_embs[stype]
        dstdata = all_embs[dtype]
        _, attn = self.convs[edges_type][str(etype)](subgraph, (srcdata, dstdata), get_attention=True)
        return attn

    def forward(self, snapshots, input_embs, get_attention=False):
        input_embs = input_embs.copy()
        hidden_embs = input_embs
        for edges_type in self.sequence:
            if edges_type != 'intra_snapshots':
                if get_attention:
                    attn_rgat = self.get_attn(snapshots, hidden_embs, edges_type, ('paper', 'is in', 'snapshot'))
                    attn_cgin = self.get_attn(snapshots, hidden_embs, edges_type, ('paper', 'cites', 'paper'))
                    attn = [attn_rgat, attn_cgin]
                hidden_embs = self.normal_forward(snapshots, hidden_embs, edges_type)
            else:
                hidden_embs = self.snapshot_forward(hidden_embs, snapshots)
        if self.activation:
            for ntype in hidden_embs.keys():
                hidden_embs[ntype] = self.activation(hidden_embs[ntype])
        if get_attention:
            return hidden_embs, attn
        else:
            return hidden_embs


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    rel_names = ['cites', 'is cited by', 'writes', 'is writen by', 'publishes', 'is published by']
    model = StochasticKLayerRGCN(768, 300, 300, rel_names)
    print(model)
