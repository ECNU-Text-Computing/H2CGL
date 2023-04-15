import argparse
import datetime

import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from layers.common_layers import MLP, CapsuleNetwork
from layers.GCN.custom_gcn import DAGIN, RGIN, RGIN_N, RGAT, DGIN, DGINM


def get_gcn_layer(encoder_type, in_feat, hidden_feat, layer, activation=None, **kwargs):
    if encoder_type == 'GCN':
        return dglnn.GraphConv(in_feat, hidden_feat, norm='right', allow_zero_in_degree=True)
    elif encoder_type == 'GIN':
        return dglnn.GINConv(MLP(in_feat, hidden_feat, activation=activation))
    elif encoder_type == 'GAT':
        nheads = kwargs.get('nheads', 4)
        out_feat = hidden_feat // nheads
        return dglnn.GATv2Conv(in_feat, out_feat, num_heads=nheads, activation=activation, allow_zero_in_degree=True)
    elif encoder_type == 'APPNP':
        k = kwargs.get('k', 5)
        alpha = kwargs.get('alpha', 0.5)
        return dglnn.APPNPConv(k=k, alpha=alpha)
    elif encoder_type == 'GCN2':
        return dglnn.GCN2Conv(in_feat, layer, allow_zero_in_degree=True, activation=activation)
    elif encoder_type == 'SAGE':
        return dglnn.SAGEConv(in_feat, hidden_feat, 'mean', activation=activation)
    elif encoder_type == 'PNA':
        return dglnn.PNAConv(in_feat, hidden_feat, ['mean', 'max', 'sum'], ['identity', 'amplification'], 2.5)
    elif encoder_type == 'DGN':
        return dglnn.DGNConv(in_feat, hidden_feat, ['dir1-av', 'dir1-dx', 'sum'], ['identity', 'amplification'], 2.5)
    elif encoder_type == 'DAGIN':
        distance_interval = kwargs.get('distance_interval', 5)
        return DAGIN(in_feat, hidden_feat, distance_interval=distance_interval, activation=activation)
    elif encoder_type == 'DGIN':
        attr_indicator = kwargs.get('attr_indicator', 'citations')
        return DGIN(in_feat, hidden_feat, attr_indicator=attr_indicator, activation=activation)
    elif encoder_type == 'DGINM':
        attr_indicator = kwargs.get('attr_indicator', 'citations')
        return DGINM(in_feat, hidden_feat, attr_indicator=attr_indicator, activation=activation)
    elif encoder_type == 'DGINP':
        attr_indicator = kwargs.get('attr_indicator', 'citations')
        return DGIN(in_feat, hidden_feat, degree_trans_method='pos',
                    attr_indicator=attr_indicator, activation=activation)
    elif encoder_type == 'DGINPM':
        attr_indicator = kwargs.get('attr_indicator', 'citations')
        return DGINM(in_feat, hidden_feat, degree_trans_method='pos',
                    attr_indicator=attr_indicator, activation=activation)
    elif encoder_type == 'RGIN':
        # print(kwargs)
        snapshot_types = kwargs.get('snapshot_types', 3)
        node_attr_types = kwargs.get('node_attr_types', [])
        return RGIN_N(in_feat, hidden_feat, snapshot_types, node_attr_types, activation=activation)
    elif encoder_type == 'CGIN':
        return dglnn.GINConv(CapsuleNetwork(in_feat, hidden_feat, activation=activation))
    elif encoder_type == 'RGAT':
        snapshot_types = kwargs.get('snapshot_types', 3)
        node_attr_types = kwargs.get('node_attr_types', [])
        print(node_attr_types)
        nheads = kwargs.get('nheads', 4)
        out_feat = hidden_feat // nheads
        return RGAT(in_feat, out_feat, nheads, snapshot_types, node_attr_types,
                    activation=activation, allow_zero_in_degree=True)


class CustomGCN(nn.Module):
    # homo single GCN encoder
    def __init__(self, encoder_type, in_feat, hidden_feat, layer, activation=None, **kwargs):
        super(CustomGCN, self).__init__()
        self.encoder_type = encoder_type
        self.fc_out = None
        # self.conv = get_gcn_layer(encoder_type, in_feat, hidden_feat, layer, activation=activation)
        if encoder_type in ['DAGIN', 'CGIN', 'DGIN']:
            self.conv = get_gcn_layer(encoder_type, in_feat, hidden_feat, layer, activation=activation)
        elif encoder_type == 'RGIN':
            snapshot_types = kwargs['snapshot_types']
            node_attr_types = kwargs['node_attr_types']
            self.conv = get_gcn_layer(encoder_type, in_feat, hidden_feat, layer, activation=activation,
                                      snapshot_types=snapshot_types, node_attr_types=node_attr_types)
        elif encoder_type == 'RGAT':
            snapshot_types = kwargs['snapshot_types']
            node_attr_types = kwargs['node_attr_types']
            self.conv = get_gcn_layer(encoder_type, in_feat, hidden_feat, layer, activation=activation,
                                      snapshot_types=snapshot_types, node_attr_types=node_attr_types)
        else:
            self.conv = get_gcn_layer(encoder_type, in_feat, hidden_feat, layer, activation=activation)

        if 'GAT' in self.encoder_type:
            self.fc_out = nn.Linear(hidden_feat, hidden_feat)
        elif self.encoder_type in ['APPNP', 'GCN2']:
            self.fc_out = nn.Linear(in_feat, hidden_feat)

    def forward(self, graph, feat, edge_weight=None, **kwargs):
        '''
        :param graph:
        :param feat: tuple of src and dst
        :param edge_weight:
        :param kwargs:
        :return: hidden_embs: {ntype: [N, out_feat]}
        '''
        # print('wip')
        # print(self.encoder_type)
        attn = None
        if self.encoder_type == 'GAT':
            feat = self.conv(graph, feat, **kwargs)
        else:
            feat = self.conv(graph, feat, edge_weight=edge_weight, **kwargs)
        if 'GAT' in self.encoder_type:
            if kwargs.get('get_attention', False):
                feat, attn = feat
            num_nodes = feat.shape[0]
            feat = feat.reshape(num_nodes, -1)
        if self.fc_out:
            feat = self.fc_out(feat)
        if attn is not None:
            return feat, attn
        else:
            return feat


class BasicHeteroGCN(nn.Module):
    def __init__(self, encoder_type, in_feat, out_feat, ntypes, etypes, layer, activation=None, **kwargs):
        super(BasicHeteroGCN, self).__init__()

        if encoder_type == 'HGT':
            nheads = kwargs.get('nheads', 4)
            self.conv = dglnn.HGTConv(in_feat, out_feat // nheads, nheads,
                                      num_ntypes=len(ntypes), num_etypes=len(etypes))
        else:
            self.conv = dglnn.HeteroGraphConv({etype: get_gcn_layer(encoder_type, in_feat, out_feat, layer,
                                                                    activation=activation) for etype in etypes})
        self.encoder_type = encoder_type
        print(self.encoder_type)
        self.fc_out = None
        if self.encoder_type == 'GAT':
            self.fc_out = dglnn.HeteroLinear({ntype: out_feat for ntype in ntypes}, out_feat)
        elif self.encoder_type in ['APPNP', 'GCN2']:
            self.fc_out = dglnn.HeteroLinear({ntype: in_feat for ntype in ntypes}, out_feat)

    def forward(self, graph, feat, edge_weight=None, **kwargs):
        '''
        :param graph:
        :param feat:
        :param edge_weight:
        :param kwargs:
        :return: hidden_embs: {ntype: [N, out_feat]}
        '''
        # print('wip')
        # print('start', [torch.isnan(feat[x]).sum() for x in feat])
        # print(self.encoder_type)
        # print(self.conv)
        if self.encoder_type == 'HGT':
            # print(feat.shape)
            feat = self.conv(graph, feat, graph.ndata[dgl.NTYPE], graph.edata[dgl.ETYPE], presorted=True)
        if self.encoder_type == 'GAT':
            feat = self.conv(graph, feat)
        else:
            # feat = self.conv(graph, feat, edge_weight=edge_weight)
            feat = self.conv(graph, feat, mod_kwargs={'edge_weight': edge_weight})
            # print('end', [torch.isnan(feat[x]).sum() for x in feat])
        if self.encoder_type == 'GAT':
            for ntype in feat:
                num_nodes = feat[ntype].shape[0]
                feat[ntype] = feat[ntype].reshape(num_nodes, -1)
        if self.fc_out:
            feat = self.fc_out(feat)
        return feat


class CustomHeteroGCN(nn.Module):
    def __init__(self, encoder_type, in_feat, out_feat, ntypes, etypes, layer, activation=None, **kwargs):
        super(CustomHeteroGCN, self).__init__()
        etype_dict = None

        if encoder_type == 'HGT':
            nheads = kwargs.get('nheads', 4)
            self.conv = dglnn.HGTConv(in_feat, out_feat // nheads, nheads,
                                      num_ntypes=len(ntypes), num_etypes=len(etypes))
        elif encoder_type in ['DAGIN', 'CGIN']:
            # print(kwargs)
            etype_dict = {}
            for etype in kwargs.get('special_edges', []):
                etype_dict[etype] = get_gcn_layer(encoder_type, in_feat, out_feat, layer, activation=activation)
            for etype in etypes:
                if etype not in etype_dict:
                    etype_dict[etype] = get_gcn_layer('GIN', in_feat, out_feat, layer, activation=activation)
        elif encoder_type == 'RGIN':
            etype_dict = {}
            for etype in kwargs.get('special_edges', []):
                snapshot_types = kwargs['snapshot_types']
                node_attr_types = kwargs['node_attr_types']
                etype_dict[etype] = get_gcn_layer(encoder_type, in_feat, out_feat, layer, activation=activation,
                                                  snapshot_types=snapshot_types, node_attr_types=node_attr_types)
            # print(etype_dict)
            for etype in etypes:
                if etype not in etype_dict:
                    etype_dict[etype] = get_gcn_layer('GIN', in_feat, out_feat, layer, activation=activation)
        elif encoder_type == 'AGIN':
            etype_dict = {}
            # DAGIN
            if 'special_edges' in kwargs:
                for etype in kwargs['special_edges']['DAGIN']:
                    etype_dict[etype] = get_gcn_layer('DAGIN', in_feat, out_feat, layer, activation=activation)
                # RGIN
                for etype in kwargs['special_edges']['RGIN']:
                    snapshot_types = kwargs['snapshot_types']
                    node_attr_types = kwargs['node_attr_types']
                    etype_dict[etype] = get_gcn_layer('RGIN', in_feat, out_feat, layer, activation=activation,
                                                      snapshot_types=snapshot_types, node_attr_types=node_attr_types)
            # print(etype_dict)
            for etype in etypes:
                if etype not in etype_dict:
                    etype_dict[etype] = get_gcn_layer('GIN', in_feat, out_feat, layer, activation=activation)
        elif 'encoder_dict' in kwargs:
            etype_dict = {}
            encoder_dict = kwargs['encoder_dict']
            for special_encoder_type in encoder_dict:
                for etype in encoder_dict[special_encoder_type]:
                    etype_dict[etype] = get_gcn_layer(special_encoder_type, in_feat, out_feat, layer,
                                                      activation=activation)
            print(etype_dict)
            for etype in etypes:
                if etype not in etype_dict:
                    etype_dict[etype] = get_gcn_layer(encoder_type, in_feat, out_feat, layer, activation=activation)
        else:
            etype_dict = {etype: get_gcn_layer(encoder_type, in_feat, out_feat, layer,
                                               activation=activation) for etype in etypes}
        if etype_dict:
            self.conv = dglnn.HeteroGraphConv(etype_dict)
        self.encoder_type = encoder_type
        # print(self.encoder_type)
        self.fc_out = None
        if self.encoder_type == 'GAT':
            self.fc_out = dglnn.HeteroLinear({ntype: out_feat for ntype in ntypes}, out_feat)
        elif self.encoder_type in ['APPNP', 'GCN2']:
            self.fc_out = dglnn.HeteroLinear({ntype: in_feat for ntype in ntypes}, out_feat)

    def forward(self, graph, feat, edge_weight=None, **kwargs):
        '''
        :param graph:
        :param feat:
        :param edge_weight:
        :param kwargs:
        :return: hidden_embs: {ntype: [N, out_feat]}
        '''
        # print('wip')
        if self.encoder_type == 'HGT':
            # print(feat.shape)
            feat = self.conv(graph, feat, graph.ndata[dgl.NTYPE], graph.edata[dgl.ETYPE], presorted=True)
        else:
            feat = self.conv(graph, feat, edge_weight=edge_weight)
        if self.encoder_type == 'GAT':
            for ntype in feat:
                num_nodes = feat[ntype].shape[0]
                feat[ntype] = feat[ntype].reshape(num_nodes, -1)
        if self.fc_out:
            feat = self.fc_out(feat)
        return feat


class StochasticKLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, ntypes, etypes, k=2, residual=False, encoder_type='GCN'):
        super().__init__()
        self.residual = residual
        self.convs = nn.ModuleList()
        # print(encoder_type, in_feat, hidden_feat)
        # self.conv_start = dglnn.HeteroGraphConv({
        #     rel: get_gcn_layer(encoder_type, in_feat, hidden_feat, activation=nn.LeakyReLU())
        #     for rel in etypes
        # })
        # self.conv_end = dglnn.HeteroGraphConv({
        #     rel: get_gcn_layer(encoder_type, hidden_feat, out_feat, activation=nn.LeakyReLU())
        #     for rel in etypes
        # })
        # self.convs.append(self.conv_start)
        # for i in range(k - 2):
        #     self.convs.append(dglnn.HeteroGraphConv({
        #         rel: get_gcn_layer(encoder_type, hidden_feat, hidden_feat, activation=nn.LeakyReLU())
        #         for rel in etypes
        #     }))
        # self.convs.append(self.conv_end)
        print(ntypes)
        self.skip_fcs = nn.ModuleList()
        self.conv_start = BasicHeteroGCN(encoder_type, in_feat, hidden_feat, ntypes, etypes, layer=1,
                                         activation=nn.LeakyReLU())
        self.skip_fc_start = dglnn.HeteroLinear({key: in_feat for key in ntypes}, hidden_feat)
        self.conv_end = BasicHeteroGCN(encoder_type, hidden_feat, out_feat, ntypes, etypes, layer=k,
                                       activation=nn.LeakyReLU())
        self.skip_fc_end = dglnn.HeteroLinear({key: hidden_feat for key in ntypes}, out_feat)
        self.convs.append(self.conv_start)
        self.skip_fcs.append(self.skip_fc_start)
        for i in range(k - 2):
            self.convs.append(BasicHeteroGCN(encoder_type, hidden_feat, hidden_feat, ntypes, etypes, layer=i + 2,
                                             activation=nn.LeakyReLU()))
            self.skip_fcs.append(dglnn.HeteroLinear({key: hidden_feat for key in ntypes}, hidden_feat))
        self.convs.append(self.conv_end)
        self.skip_fcs.append(self.skip_fc_end)

    # def forward(self, graph, x):
    #     x = self.conv_start(graph, x)
    #     x = self.conv_end(graph, x)
    #     return x
    def forward(self, graph, x):
        # print(graph.device)
        # print(x.device)
        # count = 0
        for i in range(len(self.convs)):
            # print('-' * 30 + str(count) + '-' * 30)
            # print(x)
            if self.residual:
                # x = x + conv(graph, x)
                out = self.convs[i](graph, x)
                x = self.skip_fcs[i](x)
                for key in x:
                    x[key] = x[key] + out[key]
            else:
                x = self.convs[i](graph, x)
            # print([torch.isnan(x[e]).sum() for e in x])
            # print(x)
            # count += 1
        return x

    # def forward_block(self, blocks, x):
    #     x = self.conv_start(blocks[0], x)
    #     x = self.conv_end(blocks[1], x)
    #     return x
    def forward_block(self, blocks, x):
        for i in range(len(self.convs)):
            x = self.convs[i](blocks[i], x)
        return x


class CustomRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, ntypes, etypes, k=2, residual=False, encoder_type='GCN',
                 return_list=False, **kwargs):
        super().__init__()
        self.residual = residual
        self.convs = nn.ModuleList()
        self.conv_start = CustomHeteroGCN(encoder_type, in_feat, hidden_feat, ntypes, etypes, layer=1,
                                          activation=nn.LeakyReLU(), **kwargs)
        self.conv_end = CustomHeteroGCN(encoder_type, hidden_feat, out_feat, ntypes, etypes, layer=k,
                                        activation=nn.LeakyReLU(), **kwargs)
        self.convs.append(self.conv_start)
        for i in range(k - 2):
            self.convs.append(CustomHeteroGCN(encoder_type, hidden_feat, hidden_feat, ntypes, etypes, layer=i + 2,
                                              activation=nn.LeakyReLU(), **kwargs))
        self.convs.append(self.conv_end)
        self.return_list = return_list

    def forward(self, graph, x):
        output_list = []
        for conv in self.convs:
            if self.residual:
                x = x + conv(graph, x)
            else:
                x = conv(graph, x)
            if self.return_list:
                output_list.append(x)
        return x, output_list


class HTSGCNLayer(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, ntypes, etypes, k=2, time_length=10, encoder_type='GCN',
                 return_list=False):
        super().__init__()
        # T * k layers
        self.k = k
        self.time_length = time_length
        # self.spatial_encoders = nn.ModuleList(
        #     [StochasticKLayerRGCN(in_feat, hidden_feat, out_feat, rel_names, k=k) for i in range(time_length)])
        self.spatial_encoders = nn.ModuleList(
            [CustomRGCN(in_feat, hidden_feat, out_feat, ntypes, etypes, k=k,
                        encoder_type=encoder_type, return_list=return_list) for i in range(time_length)])
        self.time_encoders = nn.ModuleList()
        for i in range(k - 1):
            self.time_encoders.append(HTEncoder(hidden_feat, hidden_feat, activation=torch.relu))
        self.time_encoders.append(HTEncoder(out_feat, out_feat, activation=torch.relu))

    # def forward(self, graph, x):
    #     x = self.conv_start(graph, x)
    #     x = self.conv_end(graph, x)
    #     return x
    def forward(self, graph_list, time_adj):
        for i in range(self.k):
            # print(i)
            for t in range(self.time_length):
                # 注意这里都是每年batch好的图
                # try:
                graph_list[t].dstdata['h'] = self.spatial_encoders[t].convs[i](graph_list[t],
                                                                               graph_list[t].srcdata['h'])
                # except Exception as e:
                #     print(e)
                #     print(graph_list[t])
                #     print(graph_list[t].batch_size)
                #     raise Exception
                # print(graph_list[t].dstdata['h']['snapshot'].shape)
            graph_list = self.time_encoders[i](graph_list, time_adj)
        return graph_list


class HTEncoder(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False, activation=None):
        super(HTEncoder, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.l_weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.r_weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.l_bias = None
        self.r_bias = None
        self.fc_out = nn.Linear(2 * out_feats, out_feats, bias=bias)
        self.activation = activation
        if bias:
            self.l_bias = nn.Parameter(torch.Tensor(out_feats))
            self.r_bias = nn.Parameter(torch.Tensor(out_feats))
        self.init_weights()

    def init_weights(self):
        if self.l_weight is not None:
            nn.init.xavier_uniform_(self.l_weight)
            nn.init.xavier_uniform_(self.r_weight)
        if self.l_bias is not None:
            nn.init.zeros_(self.l_bias)
            nn.init.zeros_(self.r_bias)

    def forward(self, snapshots, time_adj):
        # T[B], [B, 2, T, T]
        # print('wip')
        emb_list = []
        for snapshot in snapshots:
            # print(snapshot.nodes['snapshot'].data['h'].shape)
            emb_list.append(snapshot.nodes['snapshot'].data['h'])
        time_emb = torch.stack(emb_list, dim=1)  # [B, T, d_in]
        l_emb = torch.matmul(torch.matmul(time_adj[:, 0], time_emb), self.l_weight)
        r_emb = torch.matmul(torch.matmul(time_adj[:, 1], time_emb), self.r_weight)
        out_emb = torch.cat((l_emb, r_emb), dim=-1)
        if self.activation:
            out_emb = self.activation(out_emb)
        time_out = self.fc_out(out_emb)  # [B, T, d_out]
        for i in range(len(snapshots)):
            snapshots[i].nodes['snapshot'].data['h'] = time_out[:, i]
        return snapshots


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    rel_names = ['cites', 'is cited by', 'writes', 'is writen by', 'publishes', 'is published by']
    model = StochasticKLayerRGCN(768, 300, 300, rel_names)
    print(model)
