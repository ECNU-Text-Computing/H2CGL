import math

import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax

from layers.common_layers import MLP, PositionalEncoding


class DAGIN(dglnn.GINConv):
    def __init__(self,
                 # apply_func=None,
                 in_feat, hidden_feat,
                 aggregator_type='sum',
                 init_eps=0,
                 learn_eps=False,
                 activation=None,
                 distance_interval=5):
        # distance-aware GIN
        # dst and srd need have 'time'
        # paper or paper vs snapshot
        apply_func = MLP(in_feat, hidden_feat, activation=activation)
        super(DAGIN, self).__init__(apply_func, aggregator_type, init_eps, learn_eps, None)
        self.distance_interval = distance_interval
        self.distance_embeddings = nn.Embedding(2 * distance_interval + 1, in_feat * 2, padding_idx=-1)
        # self.weight_activation = nn.LeakyReLU()
        self.fc_weight = nn.Linear(in_feat * 2, 1, bias=False)

    def da_weight(self, edges):
        # print(edges.data)
        cur_distance = edges.dst['time'].squeeze(dim=-1) - edges.src['time'].squeeze(dim=-1) + self.distance_interval
        cur_distance = torch.clamp(cur_distance, min=0, max=2 * self.distance_interval - 1)
        # print(cur_distance.shape)
        distance_embs = self.distance_embeddings(cur_distance.long())
        srcdst_embs = torch.cat((edges.src['h'], edges.dst['h']), dim=-1)
        # print(distance_embs.shape)
        edge_weight = self.fc_weight(srcdst_embs + distance_embs)
        # print(snapshot_weight)
        return {'w': edge_weight}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Isomorphism Network graph_encoder.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            # aggregate_fn = fn.copy_src('h', 'm')
            # if snapshot_weight is not None:
            #     assert snapshot_weight.shape[0] == graph.number_of_edges()
            #     graph.edata['_edge_weight'] = snapshot_weight
            #     aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst

            # 额外添加的部分，根据node emb和relative distance计算节点权重，然后进行加权
            graph.apply_edges(self.da_weight)
            graph.edata['a'] = edge_softmax(graph, graph.edata['w'])
            # print(graph.edata['a'].shape)
            aggregate_fn = fn.u_mul_e('h', 'a', 'm')
            # print(graph.edata['d'])
            graph.update_all(aggregate_fn, _reducer('m', 'neigh'))

            # print(feat_dst.shape, graph.dstdata['neigh'].shape)
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class DGIN(dglnn.GINConv):
    '''
    The C-GIN that consider the is cited by highly-cited papers, single directional!
    '''
    def __init__(self,
                 # apply_func=None,
                 in_feat, hidden_feat,
                 attr_indicator='citations',
                 degree_trans_method='mlp',
                 aggregator_type='sum',
                 init_eps=0,
                 learn_eps=False,
                 activation=None):
        # degree-aware GIN
        # only consider the src node degree
        apply_func = MLP(in_feat, hidden_feat, activation=activation)
        super(DGIN, self).__init__(apply_func, aggregator_type, init_eps, learn_eps, None)
        self.indicator = attr_indicator
        self.src_trans = MLP(in_feat, in_feat, activation=activation)
        self.dst_trans = MLP(in_feat, in_feat, activation=activation)
        self.degree_trans_method = degree_trans_method
        print(self.degree_trans_method)
        if degree_trans_method == 'mlp':
            self.degree_trans = MLP(1, in_feat, dim_inner=in_feat, activation=None)
        elif degree_trans_method == 'pos':
            self.degree_trans = PositionalEncoding(in_feat, max_len=500)
        elif degree_trans_method == 'box':
            self.degree_trans = nn.Embedding(6, in_feat, padding_idx=-1)
        # self.weight_activation = nn.LeakyReLU()
        self.attn_activation = activation
        self.fc_weight = nn.Linear(in_feat, 1, bias=False)

    def get_degree(self, graph):
        if self.degree_trans_method == 'mlp':
            degree = (graph.srcdata[self.indicator].unsqueeze(dim=-1) + 1).log()
        elif self.degree_trans_method == 'pos':
            degree = torch.clamp(graph.srcdata[self.indicator].long(), min=0, max=499)
        elif self.degree_trans_method == 'box':
            degree = graph.srcdata[self.indicator].long()
        return degree

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        r"""

        Description
        -----------
        Compute Graph Isomorphism Network graph_encoder.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            # aggregate_fn = fn.copy_src('h', 'm')
            # if snapshot_weight is not None:
            #     assert snapshot_weight.shape[0] == graph.number_of_edges()
            #     graph.edata['_edge_weight'] = snapshot_weight
            #     aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst
            degree = self.get_degree(graph)
            # graph.srcdata.update({'de': self.degree_trans(degree)})
            graph.srcdata.update({'el': self.src_trans(feat_src) + self.degree_trans(degree)})
            graph.dstdata.update({'er': self.src_trans(feat_dst)})
            # graph.dstdata.update({'er': self.dst_trans(feat_dst)})

            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            graph.edata['a'] = edge_softmax(graph, self.fc_weight(self.attn_activation(graph.edata.pop('e'))))
            # print(graph.edata['a'].shape)
            aggregate_fn = fn.u_mul_e('h', 'a', 'm')
            # print(graph.edata['d'])
            graph.update_all(aggregate_fn, _reducer('m', 'neigh'))

            # print(feat_dst.shape, graph.dstdata['neigh'].shape)
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class DGINM(DGIN):
    def __init__(self, in_feat, hidden_feat,
                 threshold=5,
                 attr_indicator='citations',
                 degree_trans_method='mlp',
                 aggregator_type='sum',
                 init_eps=0,
                 learn_eps=False,
                 activation=None):
        '''
        :param in_feat:
        :param hidden_feat:
        :param threshold: e^n
        :param attr_indicator:
        :param degree_trans_method:
        :param aggregator_type:
        :param init_eps:
        :param learn_eps:
        :param activation:
        '''
        super(DGINM, self).__init__(in_feat, hidden_feat,
                 attr_indicator,
                 degree_trans_method,
                 aggregator_type,
                 init_eps,
                 learn_eps,
                 activation)
        self.threshold = threshold

    def get_degree(self, graph):
        if self.degree_trans_method == 'mlp':
            degree = torch.clamp(self.threshold - (graph.srcdata[self.indicator].unsqueeze(dim=-1) + 1).log(),
                                 min=0, max=self.threshold)
        elif self.degree_trans_method == 'pos':
            threshold = int(math.exp(self.threshold))
            degree = threshold - torch.clamp(graph.srcdata[self.indicator].long(), min=0, max=threshold)
        elif self.degree_trans_method == 'box':
            degree = graph.srcdata[self.indicator].long()
        return degree


class RGIN(dglnn.GINConv):
    def __init__(self,
                 # apply_func=None,
                 in_feat, hidden_feat, snapshot_types, edge_types,
                 # snapshot_loc,
                 aggregator_type='sum',
                 init_eps=0,
                 learn_eps=False,
                 activation=None):
        apply_func = MLP(in_feat, hidden_feat, activation=activation)
        super(RGIN, self).__init__(apply_func, aggregator_type, init_eps, learn_eps, None)
        self.snapshot_type_emb = nn.Embedding(snapshot_types + 1, in_feat * 2, padding_idx=-1)
        self.edge_type_emb = nn.Embedding(edge_types + 1, in_feat * 2, padding_idx=-1)
        self.fc_weight = nn.Linear(in_feat * 2, 1, bias=False)
        # self.snapshot_loc = snapshot_loc

    def r_weight(self, edges):
        # if self.snapshot_loc == 'src':
        #     snapshot_type_embs = edges.src['type_h']
        # else:
        #     snapshot_type_embs = edges.dst['type_h']
        snapshot_type_embs = edges.dst['type_h']
        # print(edges.data[dgl.ETYPE])
        edge_type_embs = self.edge_type_emb(edges.data[dgl.ETYPE].long())

        srcdst_embs = torch.cat((edges.src['h'], edges.dst['h']), dim=-1)
        # print(distance_embs.shape)
        edge_weight = self.fc_weight(srcdst_embs + snapshot_type_embs + edge_type_embs)
        # print(snapshot_weight)
        return {'w': edge_weight}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Isomorphism Network graph_encoder.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            # aggregate_fn = fn.copy_src('h', 'm')
            # if snapshot_weight is not None:
            #     assert snapshot_weight.shape[0] == graph.number_of_edges()
            #     graph.edata['_edge_weight'] = snapshot_weight
            #     aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst

            # if self.snapshot_loc == 'src':
            #     graph.srcdata['type_h'] = self.snapshot_type_emb(graph.srcdata['type'])
            #     print(graph.srcdata['type'])
            # else:
            #     graph.dstdata['type_h'] = self.snapshot_type_emb(graph.dstdata['type'])
            #     print(graph.dstdata['type'])
            graph.dstdata['type_h'] = self.snapshot_type_emb(graph.dstdata['type'].long())
            graph.apply_edges(self.r_weight)
            graph.edata['a'] = edge_softmax(graph, graph.edata['w'])
            # print(graph.edata['a'])
            aggregate_fn = fn.u_mul_e('h', 'a', 'm')
            # print(graph.edata['d'])
            graph.update_all(aggregate_fn, _reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class RGIN_N(RGIN):
    def __init__(self,
                 # apply_func=None,
                 in_feat, hidden_feat, snapshot_types, node_attr_types,
                 # snapshot_loc,
                 aggregator_type='sum',
                 init_eps=0,
                 learn_eps=False,
                 activation=None):
        # node type emb instead of edge type
        super(RGIN_N, self).__init__(in_feat, hidden_feat, snapshot_types, len(node_attr_types), aggregator_type, init_eps,
                                     learn_eps, activation)

        self.node_attr_type_emb = nn.ModuleDict({node_attr_type: nn.Embedding(2, in_feat * 2, padding_idx=0)
                                            for node_attr_type in node_attr_types})
        self.node_attr_types = node_attr_types

    def r_weight(self, edges):
        # if self.snapshot_loc == 'src':
        #     snapshot_type_embs = edges.src['type_h']
        # else:
        #     snapshot_type_embs = edges.dst['type_h']
        snapshot_type_embs = edges.dst['type_h']
        node_attr_type_embs = edges.src['type_h']
        # print(edges.data[dgl.ETYPE])
        # edge_type_embs = self.edge_type_emb(edges.data[dgl.ETYPE])

        # print(edges.src['h'].shape, edges.dst['h'].shape)
        srcdst_embs = torch.cat((edges.src['h'], edges.dst['h']), dim=-1)
        # print(distance_embs.shape)
        # print(srcdst_embs.shape, snapshot_type_embs.shape, node_attr_type_embs.shape)
        edge_weight = self.fc_weight(srcdst_embs + snapshot_type_embs + node_attr_type_embs)
        # print(snapshot_weight)
        return {'w': edge_weight}

    def forward(self, graph, feat, edge_weight=None):
        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            # aggregate_fn = fn.copy_src('h', 'm')
            # if snapshot_weight is not None:
            #     assert snapshot_weight.shape[0] == graph.number_of_edges()
            #     graph.edata['_edge_weight'] = snapshot_weight
            #     aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst
            # print(feat_src.shape, feat_dst.shape)

            graph.srcdata['type_h'] = torch.stack([self.node_attr_type_emb[node_attr_type](
                graph.srcdata[node_attr_type]) for node_attr_type in self.node_attr_types], dim=0).sum(dim=0)
            # print(graph.dstdata)
            graph.dstdata['type_h'] = self.snapshot_type_emb(graph.dstdata['type'].long())
            graph.apply_edges(self.r_weight)
            graph.edata['a'] = edge_softmax(graph, graph.edata['w'])
            # print(graph.edata['a'])
            aggregate_fn = fn.u_mul_e('h', 'a', 'm')
            # print(graph.edata['d'])
            graph.update_all(aggregate_fn, _reducer('m', 'neigh'))
            # print(feat_dst.shape, graph.dstdata['neigh'].shape)
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst
        
        
class RGAT(dglnn.GATv2Conv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 snapshot_types,
                 node_attr_types,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False):
        super(RGAT, self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop, negative_slope, residual,
                                   activation, allow_zero_in_degree, bias, share_weights)
        self.snapshot_type_emb = nn.Embedding(snapshot_types + 1, out_feats * num_heads, padding_idx=-1)
        self.node_attr_type_emb = nn.ModuleDict({node_attr_type: nn.Embedding(2, out_feats * num_heads, padding_idx=0)
                                                 for node_attr_type in node_attr_types})
        self.node_attr_types = node_attr_types

    def forward(self, graph, feat, get_attention=False, **kwargs):
        r"""
        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]

            # 获得snapshot_emb和ntype_emb
            graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

            src_type_h = torch.stack([self.node_attr_type_emb[node_attr_type](
                graph.srcdata[node_attr_type]) for node_attr_type in self.node_attr_types], dim=0).sum(dim=0)\
                .view(-1, self._num_heads, self._out_feats)
            # print(graph.dstdata)
            # print(src_type_h.shape, feat_src.shape)
            dst_type_h = self.snapshot_type_emb(graph.dstdata['type'].long()).view(-1, self._num_heads, self._out_feats)
            graph.srcdata.update({'etl': src_type_h})# (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'etr': dst_type_h})
            graph.apply_edges(fn.u_add_v('etl', 'etr', 'et'))

            # print(graph.edata['et'].shape)
            e = self.leaky_relu(graph.edata.pop('e') + graph.edata.pop('et'))# (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))  # (num_edge, num_heads)
            # message passing
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst



if __name__ == '__main__':
    # dglnn.GATv2Conv()
    g = dgl.graph(([0, 1, 2, 3, 2, 5, 0, 4], [1, 2, 3, 4, 0, 3, 2, 2]))
    # feat = torch.ones(6, 10)
    feat = torch.randn(6, 10)
    time_ = torch.randint(2002, 2015, (6, 1)).squeeze(dim=-1)
    print(time_)
    print(g.edges())
    g.ndata['time'] = time_
    lin = nn.Linear(10, 10)
    # conv = dglnn.GINConv(lin, 'max')
    conv = DAGIN(10, 10, 'sum')
    res = conv(g, feat)
    # print(res)

    print(len(g.edges()[0]))
    g.edata[dgl.ETYPE] = torch.randint(0, 3, (len(g.edges()[0]), 1)).squeeze(dim=-1)
    snapshot_type = torch.randint(0, 3, (6, 1)).squeeze(dim=-1)
    g.ndata['type'] = snapshot_type

    # src_conv = RGIN(10, 10, 3, 3, 'src')
    # dst_conv = RGIN(10, 10, 3, 3, 'dst')
    # res_src = src_conv(g, feat)
    # res_dst = dst_conv(g, feat)
    rconv = RGIN(10, 10, 3, 3)
    rres = rconv(g, feat)

    rconv = RGAT(10, 8, 2, 3, ['src', 'dst'])
    g.ndata['src'] = torch.randint(0, 2, (6, 1)).squeeze(dim=-1)
    g.ndata['dst'] = torch.randint(0, 2, (6, 1)).squeeze(dim=-1)
    rres = rconv(g, feat)
    print(rres.shape)
