import argparse
import datetime
import json
import logging
import time
from collections import Counter
# import multiprocessing
import joblib
import pandas as pd
from dgl import multiprocessing

import dgl
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from layers.GCN.RGCN import StochasticKLayerRGCN
from layers.common_layers import MLP
from models.base_model import BaseModel
from utilis.log_bar import TqdmToLogger


class TM:
    def __init__(self, paper_ids, hop, log_path):
        self.paper_ids = paper_ids
        self.hop = hop
        # logger = logging.getLogger()
        # logger.handlers.append(logging.FileHandler(log_path, mode='w+'))
        # self.tqdm_log = TqdmToLogger(logger, level=logging.INFO)

    def get_subgraph(self, cur_graph):
        return get_paper_ego_subgraph(self.paper_ids, hop=self.hop, graph=cur_graph, batched=True)


class TSGCN(BaseModel):
    '''The basic model which is also called H2GCN'''
    def __init__(self, vocab_size, embed_dim, num_classes, pad_index, word2vec=None, dropout=0.5, n_layers=3,
                 time_encoder='gru', time_encoder_layers=2, time_pooling='mean', hop=2,
                 time_length=10, hidden_dim=300, out_dim=256, ntypes=None, etypes=None, graph_type=None, **kwargs):
        super(TSGCN, self).__init__(vocab_size, embed_dim, num_classes, pad_index, word2vec, dropout, **kwargs)

        self.model_name = 'TSGCN'
        if graph_type:
            self.model_name += '_' + graph_type
        self.hop = hop
        self.activation = nn.Tanh()
        self.drop_en = nn.Dropout(dropout)
        self.n_layers = n_layers
        if self.n_layers != 2:
            self.model_name += '_n{}'.format(n_layers)
        self.time_length = time_length
        final_dim = out_dim
        print(ntypes)
        residual = False
        # print(residual)
        if residual:
            self.model_name += '_residual'
        self.graph_encoder = nn.ModuleList(
            [StochasticKLayerRGCN(embed_dim, hidden_dim, out_dim, ntypes, etypes, k=n_layers, residual=residual)
             for i in range(time_length)])
        if time_encoder == 'gru':
            self.time_encoder = nn.GRU(input_size=out_dim, hidden_size=out_dim, dropout=dropout, batch_first=True,
                                       bidirectional=True, num_layers=2)
            final_dim = 2 * out_dim
        elif time_encoder == 'self-attention':
            self.time_encoder = nn.Sequential(*[
                nn.TransformerEncoderLayer(d_model=out_dim, nhead=out_dim // 64, dim_feedforward=2 * out_dim)
                for i in range(time_encoder_layers)])
            # final_dim = 2 * out_dim
        self.time_pooling = time_pooling
        # self.fc = MLP(dim_in=final_dim, dim_inner=final_dim // 2, dim_out=num_classes, num_layers=2)
        self.fc_reg = MLP(dim_in=final_dim, dim_inner=final_dim // 2, dim_out=1, num_layers=2)
        self.fc_cls = MLP(dim_in=final_dim, dim_inner=final_dim // 2, dim_out=num_classes, num_layers=2)


    def load_graphs(self, graphs):
        '''
        {
            'data': {phase: None},
            'node_trans': json.load(open(self.data_cat_path + 'sample_node_trans.json', 'r')),
            'time': {'train': train_time, 'val': val_time, 'test': test_time},
            'all_graphs': graphs,
            'selected_papers': {phase: [list]},
            'all_trans_index': dict,
            'all_ego_graphs': ego_graphs
        }
        '''
        print(graphs['data'])
        logger = logging.getLogger()
        # LOG = logging.getLogger(__name__)
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        graphs['all_trans_index'] = json.load(open(graphs['graphs_path'] + '_trans.json', 'r'))
        # ego_graphs
        graphs['all_ego_graphs'] = {}
        for pub_time in tqdm(graphs['all_graphs'], file=tqdm_out, mininterval=30):
            graphs['all_ego_graphs'][pub_time] = joblib.load(
                graphs['graphs_path'] + '_{}.job'.format(pub_time))

        phase_dict = {phase: [] for phase in graphs['data']}
        for phase in graphs['data']:
            selected_papers = [graphs['node_trans']['paper'][paper] for paper in graphs['selected_papers'][phase]]
            trans_index = dict(zip(selected_papers, range(len(selected_papers))))
            graphs['selected_papers'][phase] = selected_papers
            graphs['data'][phase] = [None, trans_index]

        time2phase = {pub_time: [phase for phase in graphs['data']
                                 if (pub_time <= graphs['time'][phase]) and (pub_time > graphs['time'][phase] - 10)]
                      for pub_time in graphs['all_graphs']}
        for t in sorted(list(time2phase.keys())):
            cur_snapshot = graphs['all_ego_graphs'][t]
            start_time = time.time()
            for phase in time2phase[t]:
                batched_graphs = [cur_snapshot[graphs['all_trans_index'][str(paper)]]
                                  for paper in graphs['selected_papers'][phase]]
                cur_emb_graph = graphs['all_graphs'][t]
                # unbatched_graphs = dgl.unbatch(batched_graphs)
                unbatched_graphs = batched_graphs
                logging.info('unbatching done! {:.2f}s'.format(time.time() - start_time))
                for graph in tqdm(unbatched_graphs, file=tqdm_out, mininterval=30):
                    for ntype in graph.ntypes:
                        cur_index = graph.nodes[ntype].data[dgl.NID]
                        graph.nodes[ntype].data['h'] = cur_emb_graph.nodes[ntype].data['h'][cur_index].detach()
                phase_dict[phase].append(unbatched_graphs)

            logging.info('-' * 30 + str(t) + ' done' + '-' * 30)
            del cur_snapshot
            del graphs['all_graphs'][t]
            del graphs['all_ego_graphs'][t]

        for phase in phase_dict:
            graphs['data'][phase][0] = phase_dict[phase]

        del phase_dict
        del graphs['all_graphs']
        del graphs['all_ego_graphs']
        return graphs

    def deal_graphs(self, graphs, data_path=None, path=None, log_path=None):
        tqdm_out = None
        if log_path:
            logging.basicConfig(level=logging.INFO,
                                filename=log_path,
                                filemode='w+',
                                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                force=True)
            logger = logging.getLogger()
            # LOG = logging.getLogger(__name__)
            tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        phases = list(graphs['data'].keys())
        # dealt_graphs = {'data': {}, 'time': graphs['time'], 'node_trans': graphs['node_trans'], 'all_graphs': graphs['all_graphs']}
        dealt_graphs = {'data': {}, 'time': graphs['time'], 'node_trans': graphs['node_trans'],
                        'all_graphs': None}
        split_data = torch.load(data_path)
        all_selected_papers = [graphs['node_trans']['paper'][paper] for paper in split_data['test'][0]]
        logging.info('all count: {}'.format(str(len(all_selected_papers))))
        # all_snapshots = {}
        all_trans_index = dict(zip(all_selected_papers, range(len(all_selected_papers))))
        pub_times = list(graphs['all_graphs'].keys())
        # del graphs['all_graphs']
        del graphs['data']
        # print('not get ego_subgraph for saving time')
        for pub_time in pub_times:
            # all_snapshots[pub_time] = get_paper_ego_subgraph(all_selected_papers, self.hop,
            #                                                  graphs['all_graphs'][pub_time],
            #                                                  batched=False, tqdm_log=tqdm_out)
            cur_snapshot = get_paper_ego_subgraph(all_selected_papers, self.hop,
                                                  graphs['all_graphs'][pub_time],
                                                  batched=False, tqdm_log=tqdm_out)
            # if path:
            #     torch.save(cur_snapshot, path + '_' + str(pub_time))
            if path:
                joblib.dump(cur_snapshot, path + '_' + str(pub_time) + '.job')
            # del graphs['all_graphs'][pub_time]

        if path:
            json.dump(all_trans_index, open(path + '_trans.json', 'w+'))

        # all_trans_index = json.load()
        del graphs['all_graphs']

        joblib.dump([], path)
        return dealt_graphs

    def forward(self, content, lengths, masks, ids, graph, times, **kwargs):
        # graphs T个batched_graph
        # start_time = time.time()
        # print(start_time)
        # snapshots = [batch.to(self.device) for batch in snapshots]
        # print(time.time() - start_time)
        batched_graphs_list, trans_index = graph
        # graphs_list = dgl.unbatch(batched_graphs)
        snapshots = []
        for batched_graphs in batched_graphs_list:
            temp_snapshots = []
            # temp_graphs = dgl.unbatch(batched_graphs)
            temp_graphs = batched_graphs
            for paper in ids:
                temp_snapshots.append(temp_graphs[trans_index[paper.item()]])
            snapshots.append(dgl.batch(temp_snapshots).to(self.device))
        # print(time.time() - start_time)

        all_snapshots = []
        for t in range(self.time_length):
            cur_snapshot = snapshots[t]
            out_emb = self.graph_encoder[t](cur_snapshot, cur_snapshot.srcdata['h'])
            cur_snapshot.dstdata['h'] = out_emb
            sum_readout = dgl.readout_nodes(cur_snapshot, 'h', op='sum', ntype='paper')
            all_snapshots.append(sum_readout)
        all_snapshots = torch.stack(all_snapshots, dim=1)  # [B, T, d_out]
        # print(all_snapshots.shape)
        time_out = self.time_encoder(all_snapshots)
        if type(time_out) != torch.Tensor:
            time_out = time_out[0]
        # print(time_out.shape)
        if self.time_pooling == 'mean':
            final_out = time_out.mean(dim=1)
        elif self.time_pooling == 'sum':
            final_out = time_out.sum(dim=1)
        elif self.time_pooling == 'max':
            final_out = time_out.max(dim=1)[0]
        # print(final_out.shape)
        output_reg = self.fc_reg(final_out)
        output_cls = self.fc_cls(final_out)
        return output_reg, output_cls


def get_paper_ego_subgraph(batch_ids, hop, graph, batched=True, tqdm_log=None):
    '''
    get ego batch_graphs of current time
    target-centric graphs extracted from complete graph
    :param batch_ids:
    :param hop:
    :param graph:
    :return:
    '''
    oids = graph.nodes['paper'].data[dgl.NID].numpy().tolist()
    cids = graph.nodes('paper').numpy().tolist()
    oc_trans = dict(zip(oids, cids))
    trans_papers = [oc_trans.get(paper, None) for paper in batch_ids]
    graph.nodes['paper'].data['citations'] = graph.in_degrees(etype='cites')

    paper_subgraph = dgl.node_type_subgraph(graph, ['paper'])
    all_subgraphs = []
    single_subgraph = SingleSubgraph(paper_subgraph, graph, hop)
    for paper in tqdm(trans_papers, file=tqdm_log, mininterval=30):
        # for paper in tqdm(trans_papers):
        cur_graph = single_subgraph.get_single_subgraph(paper)
        for ntype in cur_graph.ntypes:
            if 'h' in cur_graph.nodes[ntype].data:
                del cur_graph.nodes[ntype].data['h']
        all_subgraphs.append(cur_graph)
    # logging.info('cur year done!')
    if batched:
        return dgl.batch(all_subgraphs)
    else:
        return all_subgraphs


class SingleSubgraph:
    def __init__(self, paper_subgraph, origin_graph, hop, citation='global'):
        # one-hop instead
        self.paper_subgraph = paper_subgraph
        # self.origin_graph = origin_graph
        self.origin_graph = dgl.node_type_subgraph(origin_graph, ntypes=['paper', 'journal', 'author'])
        self.hop = 1
        self.citation = citation
        self.max_nodes = 100
        # self.oids = self.paper_subgraph.nodes['paper'].data[dgl.NID]
        self.citations = self.paper_subgraph.nodes['paper'].data['citations']
        self.time = self.paper_subgraph.nodes['paper'].data['time']

    def get_single_subgraph(self, paper):
        if paper is not None:
            ref_hop_papers = self.paper_subgraph.in_edges(paper, etype='is cited by')[0].numpy().tolist()
            if len(ref_hop_papers) > self.max_nodes:
                temp_df = pd.DataFrame(
                    data={
                        'id': ref_hop_papers,
                        'citations': self.citations[ref_hop_papers].numpy(),
                        'time': self.time[ref_hop_papers].squeeze(dim=-1).numpy()})
                ref_hop_papers = list(temp_df.sort_values(by=['time', 'citations'],
                                                          ascending=False).head(self.max_nodes)['id'].tolist())
            cite_hop_papers = self.paper_subgraph.in_edges(paper, etype='cites')[0].numpy().tolist()
            if len(cite_hop_papers) > self.max_nodes:
                temp_df = pd.DataFrame(
                    data={
                        'id': cite_hop_papers,
                        'citations': self.citations[cite_hop_papers].numpy(),
                        'time': self.time[cite_hop_papers].squeeze(dim=-1).numpy()})
                cite_hop_papers = list(temp_df.sort_values(by=['time', 'citations'],
                                                           ascending=False).head(self.max_nodes)['id'].tolist())

            # print(len(ref_hop_papers), len(cite_hop_papers))
            # print(len(cite_hop_papers))
            k_hop_papers = list(set(ref_hop_papers + cite_hop_papers + [paper]))
            # print(ref_hop_papers, cite_hop_papers)
            # print(k_hop_papers)
            # print(self.origin_graph)
            temp_graph = dgl.in_subgraph(self.origin_graph, nodes={'paper': k_hop_papers}, relabel_nodes=True)
            journals = temp_graph.nodes['journal'].data[dgl.NID]
            authors = temp_graph.nodes['author'].data[dgl.NID]
            cur_graph = dgl.node_subgraph(self.origin_graph,
                                          nodes={'paper': k_hop_papers, 'journal': journals, 'author': authors})

            # 修改了indicator
            ref_set = set(ref_hop_papers)
            cite_set = set(cite_hop_papers)
            cur_graph.nodes['paper'].data['is_ref'] = torch.zeros(len(k_hop_papers), dtype=torch.long)
            cur_graph.nodes['paper'].data['is_cite'] = torch.zeros(len(k_hop_papers), dtype=torch.long)
            cur_graph.nodes['paper'].data['is_target'] = torch.zeros(len(k_hop_papers), dtype=torch.long)
            oids = cur_graph.nodes['paper'].data[dgl.NID].numpy().tolist()
            id_trans = dict(zip(oids, range(len(oids))))
            ref_idx = [id_trans[paper] for paper in ref_set]
            cite_idx = [id_trans[paper] for paper in cite_set]
            cur_graph.nodes['paper'].data['is_ref'][ref_idx] = 1
            cur_graph.nodes['paper'].data['is_cite'][cite_idx] = 1
            cur_graph.nodes['paper'].data['is_target'][id_trans[paper]] = 1
        else:
            cur_graph = dgl.node_subgraph(self.origin_graph, nodes={'paper': []})
            cur_graph.nodes['paper'].data['is_ref'] = torch.zeros(0, dtype=torch.long)
            cur_graph.nodes['paper'].data['is_cite'] = torch.zeros(0, dtype=torch.long)
            cur_graph.nodes['paper'].data['is_target'] = torch.zeros(0, dtype=torch.long)
        return cur_graph


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    rel_names = ['cites', 'is cited by', 'writes', 'is writen by', 'publishes', 'is published by']
    # vocab_size, embed_dim, num_classes, pad_index

    graphs = {}
    graph_name = 'graph_sample_feature_vector'
    time_list = range(2002, 2012)
    for i in range(len(time_list)):
        graphs[i] = torch.load('../data/pubmed/' + graph_name + '_' + str(time_list[i]))

    graph_data = {
        'data': graphs,
        'node_trans': json.load(open('../data/pubmed/' + 'sample_node_trans.json', 'r'))
    }
    print(graph_data['data'])
    print(graph_data['data'][0].nodes('paper'))
    # print(graph_data['node_trans']['paper'])
    reverse_paper_dict = dict(zip(graph_data['node_trans']['paper'].values(), graph_data['node_trans']['paper'].keys()))
    # print(reverse_paper_dict)
    citation_dict = json.load(open('../data/pubmed/sample_citation_accum.json'))
    valid_ids = set([reverse_paper_dict[idx.item()] for idx in graph_data['data'][0].nodes('paper')])
    useful_ids = [idx for idx in valid_ids if idx in citation_dict]
    print(len(useful_ids))
    print(useful_ids[:10])

    # sample_ids = ['3266430', '2790711', '2747541', '3016217', '7111196']
    # trans_ids = [graph_data['node_trans']['paper'][idx] for idx in sample_ids]
    # print(trans_ids)
    # x = torch.randint(100, (5, 20)).cuda()
    # lens = [20] * 5
    # print(x)
    # model = TSGCN(100, 768, 0, 0, etypes=etypes).cuda()
    # output = model(x, lens, None, trans_ids, graph_data['data'])
    # print(output)
    # print(output.shape)

    sample_ids = [4724, 12968, 8536, 9558, None]
    cur_graph = graph_data['data'][9]
    print(cur_graph.nodes['paper'].data[dgl.NID][[68, 2855, 377, 845]])
    batch_graphs = get_paper_ego_subgraph(sample_ids, 2, cur_graph, batched=False)
    print(batch_graphs)
    # print(batch_graphs.batch_size)
    # snapshots = []
    # for i in range(len(time_list)):
    #     snapshots.append(get_paper_ego_subgraph(sample_ids, 2, graph_data['data'][i]))
    # print(snapshots)
    # graph_encoder = StochasticKLayerRGCN(300, 128, 128, etypes, k=2)
    # cur_snapshot = snapshots[0]
    # out_emb = graph_encoder(cur_snapshot, cur_snapshot.srcdata['h'])
    # cur_snapshot.dstdata['h'] = out_emb

    # sum_readout = dgl.readout_nodes(cur_snapshot, 'h', op='max', ntype='paper')
    # print([vector.norm(2) for vector in sum_readout])
    # print(sum_readout.shape)

    # model = TSGCN(0, 300, 1, 0, etypes=etypes, time_encoder='gru').cuda()
    # # print(model(None, None, None, None, snapshots))
    # g1 = torch.load('../data/pubmed/graph_sample_feature_vector_2002')
    # g2 = torch.load('../data/pubmed/graph_sample_feature_vector_2003')
    # g3 = torch.load('../data/pubmed/graph_sample_feature_vector_2004')
    # graphs = [g1, g2, g3]
    # # sub_g = g1.edge_type_subgraph(['is cited by'])
    # # print(sub_g)
    # # print(sub_g.ndata[dgl.NID])
    # # print(sub_g.ndata['h'].shape)
    # node_trans = json.load(open('../data/pubmed/sample_node_trans.json'))
    # graphs_dict = {
    #     'data': {'train': graphs},
    #     'node_trans': node_trans
    # }
    # graphs = model.deal_graphs(graphs_dict, '../data/pubmed/split_data', '../checkpoints/pubmed/TSGCN')
