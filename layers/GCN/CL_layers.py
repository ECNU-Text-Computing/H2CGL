import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import dgl
from tqdm import tqdm

from layers.common_layers import MLP
from utilis.scripts import add_new_elements


class CommonCLLayer(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_feats=None, tau=0.4):
        super(CommonCLLayer, self).__init__()
        if hidden_feats is None:
            hidden_feats = in_feats
        # print(in_feats, out_feats, hidden_feats)
        self.proj_head = MLP(in_feats, out_feats, dim_inner=hidden_feats, activation=nn.ELU())
        self.tau = tau

    def forward(self, z1, z2):
        h1, h2 = F.normalize(self.proj_head(z1)), F.normalize(self.proj_head(z2))
        s12 = cos_sim(h1, h2, self.tau, norm=False)
        if s12.shape[0] == 0:
            s12 = torch.tensor([1]).reshape(1, 1).to(h1.device)
        s21 = s12

        # compute InfoNCE
        loss12 = -torch.log(s12.diag()) + torch.log(s12.sum(1))
        loss21 = -torch.log(s21.diag()) + torch.log(s21.sum(1))
        L_node = (loss12 + loss21) / 2


        return L_node.mean()


class ProjectionHead(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_feats=None):
        super(ProjectionHead, self).__init__()
        if hidden_feats is None:
            hidden_feats = in_feats
        self.proj_head = MLP(in_feats, out_feats, dim_inner=hidden_feats, activation=nn.ELU())

    def forward(self, z):
        return F.normalize(self.proj_head(z))


def simple_cl(z1, z2, tau, valid_lens=None):
    h1, h2 = z1, z2
    s12 = cos_sim(h1, h2, tau, norm=False)
    if s12.shape[0] == 0:
        s12 = torch.tensor([1]).reshape(1, 1).to(h1.device)
    s21 = s12

    # compute InfoNCE
    if s12.dim() > 2:
        # print(s12.shape)
        num_node = s12.shape[-1]
        # loss12 = -torch.log(torch.flatten(torch.diagonal(s12, dim2=-2, dim1=-1))) + \
        #          torch.log(s12.reshape(-1, num_node).sum(1))
        # loss21 = -torch.log(torch.flatten(torch.diagonal(s21, dim2=-2, dim1=-1))) + \
        #          torch.log(s21.reshape(-1, num_node).sum(1))
        # print(loss21.shape)
        loss12 = -torch.log(torch.diagonal(s12, dim2=-2, dim1=-1)) + \
                 torch.log(s12.sum(-1))
        loss21 = -torch.log(torch.diagonal(s21, dim2=-2, dim1=-1)) + \
                 torch.log(s21.sum(-1))
    else:
        loss12 = -torch.log(s12.diag()) + torch.log(s12.sum(1))
        loss21 = -torch.log(s21.diag()) + torch.log(s21.sum(1))
    L_node = (loss12 + loss21) / 2

    if type(valid_lens) == torch.Tensor:
        # print(valid_lens)
        # print(L_node.sum(dim=-1).shape)
        L_node = L_node.sum(dim=-1) / valid_lens
        # print(L_node.mean())
    return L_node.mean()

def combined_sparse(matrices):
    indices = []
    values = torch.cat([torch.flatten(matrix) for matrix in matrices])
    # print(values)
    x, y = 0, 0
    for matrix in matrices:
        cur_x, cur_y = matrix.shape
        for i in range(x, x + cur_x):
            for j in range(y, y + cur_y):
                indices.append([i, j])
        # values = torch.cat((values, torch.flatten(matrix)))
        x += cur_x
        y += cur_y
    # print(len(indices))
    # print(indices)
    # print(values.shape)
    return torch.sparse_coo_tensor(indices=torch.tensor(indices, device=matrices[0].device).T,
                                   values=values, size=(x, y), requires_grad=True)


def cos_sim(x1, x2, tau: float, norm: bool = False):
    if x1.is_sparse:
        if norm:
            return torch.softmax(torch.sparse.mm(x1, x2.transpose(0, 1)).to_dense() / tau, dim=1)
        else:
            return torch.exp(torch.sparse.mm(x1, x2.transpose(0, 1)).to_dense() / tau)
    elif x1.dim() > 2:
        result = x1 @ x2.transpose(-2, -1) / tau
        if norm:
            return torch.softmax(result, dim=-1)
        else:
            return torch.exp(result)
    else:
        if norm:
            return torch.softmax(x1 @ x2.T / tau, dim=1)
        else:
            return torch.exp(x1 @ x2.T / tau)


def RBF_sim(x1, x2, tau: float, norm: bool = False):
    xx1, xx2 = torch.stack(len(x2) * [x1]), torch.stack(len(x1) * [x2])
    sub = xx1.transpose(0, 1) - xx2
    R = (sub * sub).sum(dim=2)
    if norm:
        return F.softmax(-R / (tau * tau), dim=1)
    else:
        return torch.exp(-R / (tau * tau))


def augmenting_graphs(graphs, aug_configs, aug_methods=(('attr_drop',), ('edge_drop',))):
    '''
    :param aug_configs:
    :param graphs:
    :param aug_methods:
    :return:
    '''
    g1_list = []
    g2_list = []
    for graph in graphs:
        # g1, g2 = copy.deepcopy(graph), copy.deepcopy(graph)
        g1, g2 = add_new_elements(graph), add_new_elements(graph)
        for aug_method in aug_methods[0]:
            cur_method = eval(aug_method)
            g1 = cur_method(g1, aug_configs[0][aug_method]['prob'], aug_configs[0][aug_method].get('types', None))
        for aug_method in aug_methods[1]:
            cur_method = eval(aug_method)
            g2 = cur_method(g2, aug_configs[1][aug_method]['prob'], aug_configs[1][aug_method].get('types', None))
        g1_list.append(g1)
        g2_list.append(g2)

    del graphs
    return g1_list, g2_list


def augmenting_graphs_ctsgcn(graphs, aug_configs, aug_methods=(('attr_drop',), ('edge_drop',)), time_length=10):
    '''
    :param time_length:
    :param aug_configs:
    :param graphs:
    :param aug_methods:
    :return:
    '''
    g1_list = []
    g2_list = []
    for graph in graphs:
        # g1, g2 = copy.deepcopy(graph), copy.deepcopy(graph)
        g1, g2 = add_new_elements(graph), add_new_elements(graph)
        for t in range(time_length):
            # g1
            for aug_method in aug_methods[0]:
                selected_nodes = {}
                for ntype in graph.ntypes:
                    selected_nodes[ntype] = g1.nodes[ntype].data['snapshot_idx'] == t
                if torch.sum(selected_nodes['paper']).item() > 0:
                    cur_method = eval(aug_method)
                    g1 = cur_method(g1, aug_configs[0][aug_method]['prob'], aug_configs[0][aug_method].get('types', None),
                                    selected_nodes)
            # g2
            for aug_method in aug_methods[1]:
                selected_nodes = {}
                for ntype in graph.ntypes:
                    selected_nodes[ntype] = g2.nodes[ntype].data['snapshot_idx'] == t
                if torch.sum(selected_nodes['paper']).item() > 0:
                    cur_method = eval(aug_method)
                    g2 = cur_method(g2, aug_configs[1][aug_method]['prob'], aug_configs[1][aug_method].get('types', None),
                                    selected_nodes)
        g1_list.append(g1)
        g2_list.append(g2)

    del graphs
    return g1_list, g2_list


def augmenting_graphs_single(graphs, aug_config, aug_methods=(('attr_drop',), ('edge_drop',))):
    '''
    :param aug_configs:
    :param graphs:
    :param aug_methods:
    :return:
    '''
    graph_list = []
    for graph in graphs:
        # dealt_graph = copy.deepcopy(graph)
        dealt_graph = add_new_elements(graph)
        for aug_method in aug_methods:
            cur_method = eval(aug_method)
            dealt_graph = cur_method(dealt_graph, aug_config[aug_method]['prob'],
                                     aug_config[aug_method].get('types', None))
        graph_list.append(dealt_graph)
    return graph_list


def augmenting_graphs_single_ctsgcn(graphs, aug_config, aug_methods=(('attr_drop',), ('edge_drop',)), time_length=10):
    '''
    :param aug_configs:
    :param graphs:
    :param aug_methods:
    :return:
    '''
    graph_list = []
    for graph in graphs:
        # dealt_graph = copy.deepcopy(graph)
        dealt_graph = add_new_elements(graph)
        for aug_method in aug_methods:
            cur_method = eval(aug_method)
            for t in range(time_length):
                selected_nodes = {}
                for ntype in graph.ntypes:
                    selected_nodes[ntype] = graph.nodes[ntype].data['snapshot_idx'] == t
                if torch.sum(selected_nodes['paper']).item() == 0:
                    continue
                dealt_graph = cur_method(dealt_graph, aug_config[aug_method]['prob'],
                                         aug_config[aug_method].get('types', None), selected_nodes)
        graph_list.append(dealt_graph)
    return graph_list


def attr_mask(graph, prob, ntypes=None, selected_nodes=None):
    if ntypes is None:
        ntypes = graph.ntypes
    if selected_nodes:
        for ntype in ntypes:
            emb_dim = graph.nodes[ntype].data['h'].shape[-1]
            p = torch.bernoulli(torch.tensor([1 - prob] * emb_dim)).to(torch.bool)
            graph.nodes[ntype].data['h'][selected_nodes[ntype]][:, p] = 0.
    else:
        for ntype in ntypes:
            emb_dim = graph.nodes[ntype].data['h'].shape[-1]
            p = torch.bernoulli(torch.tensor([1 - prob] * emb_dim)).to(torch.bool)
            graph.nodes[ntype].data['h'][:, p] = 0.
    return graph


def attr_drop(graph, prob, ntypes=None, selected_nodes=None):
    if ntypes is None:
        ntypes = graph.ntypes
    if selected_nodes:
        for ntype in ntypes:
            node_count = graph.nodes[ntype].data['h'][selected_nodes[ntype]].shape[0]
            emb_dim = graph.nodes[ntype].data['h'].shape[-1]
            p = np.random.rand(node_count)
            drop_nodes = torch.where(selected_nodes[ntype])[0][np.where(p <= prob)[0]]
            graph.nodes[ntype].data['h'][drop_nodes, :] = torch.zeros(len(drop_nodes), emb_dim)
    else:
        for ntype in ntypes:
            node_count = graph.nodes[ntype].data['h'].shape[0]
            emb_dim = graph.nodes[ntype].data['h'].shape[-1]
            p = np.random.rand(node_count)
            drop_nodes = np.where(p <= prob)[0]
            graph.nodes[ntype].data['h'][drop_nodes, :] = torch.zeros(len(drop_nodes), emb_dim)
    return graph


def edge_drop(graph, prob, etypes=None, selected_nodes=None):
    # graph = copy.deepcopy(graph)
    if etypes is None:
        etypes = graph.canonical_etypes
    if selected_nodes:
        # print('wip')
        for etype in etypes:
            dst_type = etype[-1]
            dst_nodes = set(graph.nodes(dst_type)[selected_nodes[dst_type]].numpy().tolist())
            # print(graph.edges(form='all', etype=etype, order='eid'))
            src, dst, edges = graph.edges(form='all', etype=etype, order='eid')
            dst_index = []
            # print(edges)
            for i in range(len(dst)):
                if dst[i].item() in dst_nodes:
                    dst_index.append(i)
            edges = edges[dst_index]
            # print(dst_nodes, dst_index)

            p = np.random.rand(edges.shape[0])
            # print(edges[np.where(p < 0.2)])
            drop_edges = edges[np.where(p <= prob)]
            graph.remove_edges(drop_edges, etype=etype)
    else:
        for etype in etypes:
            edges = graph.edges(form='eid', etype=etype)
            p = np.random.rand(edges.shape[0])
            # print(edges[np.where(p < 0.2)])
            drop_edges = edges[np.where(p <= prob)]
            graph.remove_edges(drop_edges, etype=etype)
            # print(graph.edges(form='eid', etype='cites'))
    return graph

def node_drop(graph, prob, ntypes=None, selected_nodes=None):
    # drop nodes according to the citation, so just drop the paper, not including other nodes and the target node
    selected_index = (1 - graph.nodes['paper'].data['is_target']).to(torch.bool)
    if selected_nodes:
        selected_nodes = selected_nodes['paper']  # 这里输入的是bool
        selected_index = selected_index & selected_nodes

    count = torch.sum(selected_index).item()
    # print(count)
    if count > 0:
        drop_rate = count * prob
        cur_p = torch.softmax(graph.nodes['paper'].data['citations'][selected_index].float(), dim=-1).numpy() * drop_rate
        drop_indicator = np.random.rand(count) < cur_p
        drop_nodes = torch.where(selected_index)[0][drop_indicator]
        graph.remove_nodes(drop_nodes, ntype='paper')
    return graph


if __name__ == '__main__':
    cl = CommonCLLayer(128, 64)
    z1 = torch.randn(10, 128)
    z2 = torch.randn(10, 128)
    print(cl(z1, z2))

    print(eval('attr_drop'))
