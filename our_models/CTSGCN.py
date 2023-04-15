import copy
import random
import time
from collections import defaultdict
# import multiprocessing
import dgl
import numpy as np
import pandas as pd
import torch
# import winnt
from dgl.nn.functional import edge_softmax

from layers.GCN.CL_layers import augmenting_graphs_single, augmenting_graphs, augmenting_graphs_ctsgcn
from layers.GCN.RGCN import HTSGCNLayer, CustomRGCN
from our_models.TSGCN import *
from utilis.log_bar import TqdmToLogger
from utilis.scripts import add_new_elements, get_norm_adj, IndexDict
import dgl.nn.pytorch as dglnn


class CTSGCN(BaseModel):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_index, word2vec=None, dropout=0.5, n_layers=3,
                 pred_type='paper', time_pooling='mean', hop=2, linear_before_gcn=False, encoder_type='GCN',
                 time_length=10, hidden_dim=300, out_dim=256, ntypes=None, etypes=None, graph_type=None,
                 gcn_out='last', hn=None, hn_method=None, max_time_length=5, **kwargs):
        super(CTSGCN, self).__init__(vocab_size, embed_dim, num_classes, pad_index, word2vec, dropout, **kwargs)
        self.model_name = 'CTSGCN_{}_{}_{}_n{}'.format(pred_type, encoder_type, gcn_out, n_layers)
        if graph_type:
            self.model_name += '_' + graph_type
        self.hop = hop
        self.pred_type = pred_type
        print('--cur pred type--:', pred_type)
        self.activation = nn.Tanh()
        self.drop_en = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.time_length = time_length
        print('--cur time length--:', self.time_length)
        self.max_time_length = max_time_length
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.final_dim = out_dim
        self.ntype_fc = None
        self.cut_distance = 5
        self.cl_type = 'none'
        if linear_before_gcn:
            self.ntype_fc = dglnn.HeteroLinear({ntype: embed_dim for ntype in ntypes}, self.hidden_dim)
            embed_dim = self.hidden_dim
            self.model_name += '_lbg'
        self.encoder_type = encoder_type
        return_list = False if gcn_out == 'last' else True
        self.gcn_out = gcn_out
        self.embed_dim = embed_dim
        self.graph_encoder = CustomRGCN(embed_dim, hidden_dim, out_dim, ntypes, etypes,
                                        k=n_layers, encoder_type=encoder_type, return_list=return_list)
        # self.graph_encoder = CustomRGCN(embed_dim, hidden_dim, out_dim, ntypes, etypes,
        #                                 k=n_layers, encoder_type=encoder_type, return_list=return_list)
        # self.time_encoder = None
        self.ntypes = ntypes

        # snapshot time emb
        self.snapshot_embedding = dglnn.HeteroEmbedding({ntype: time_length for ntype in ntypes}, embed_dim)

        self.time_pooling = time_pooling
        # reg fc not for class
        self.fc_reg = MLP(dim_in=self.final_dim, dim_inner=self.final_dim // 2, dim_out=1, num_layers=2)
        self.fc_cls = MLP(dim_in=self.final_dim, dim_inner=self.final_dim // 2, dim_out=num_classes, num_layers=2)
        self.hn = hn
        self.hn_method = hn_method
        print('>>>hard negative samples: {}<<<'.format(self.hn))
        print('>>>hard negative method: {}<<<'.format(self.hn_method))

        if self.max_time_length != self.time_length:
            self.model_name += '_t{}'.format(self.time_length)
        if self.hn:
            self.model_name += '_hn{}'.format(self.hn)
        if self.hn_method:
            self.model_name += '_' + hn_method

        if gcn_out == 'concat':
            # self.fc_out = nn.Linear(n_layers * out_dim, final_dim)
            # self.fc_out = MLP(n_layers * out_dim, self.final_dim, activation=nn.LeakyReLU())
            # self.fc_out = dglnn.HeteroLinear({ntype: n_layers * out_dim for ntype in self.ntypes}, self.final_dim)
            self.fc_out = nn.ModuleDict({ntype: MLP(n_layers * out_dim, self.final_dim, activation=nn.LeakyReLU())
                                         for ntype in self.ntypes})

        self.norm = None
        self.dn = None
        oargs = kwargs.get('oargs', None)
        if oargs:
            if 'bn' in oargs:
                self.norm = nn.BatchNorm1d(self.final_dim)
                self.norm_type = 'bn'
                self.model_name += '_' + self.norm_type
            elif 'ln' in oargs:
                self.norm = nn.LayerNorm(self.final_dim)
                self.norm_type = 'ln'
                self.model_name += '_' + self.norm_type
            if 'dn' in oargs:
                self.dn = nn.Dropout(dropout)
                self.model_name += '_' + 'dn'
            if '=' in oargs:
                self.model_name += '_{}_{}_{}'.format(self.embed_dim, self.hidden_dim, self.out_dim)

    def get_time_subgraph(self, graph):
        if self.time_length != self.max_time_length:
            time_dist = self.max_time_length - self.time_length
            # print(graph.nodes['paper'].data[dgl.NID])
            sub_dict = {}
            for ntype in graph.ntypes:
                cur_index = torch.where(graph.nodes[ntype].data['snapshot_idx'] >= time_dist)[0]
                sub_dict[ntype] = graph.nodes(ntype)[cur_index]
            graph = dgl.node_subgraph(graph, nodes=sub_dict, store_ids=False)
            for ntype in graph.ntypes:
                graph.nodes[ntype].data['snapshot_idx'] = graph.nodes[ntype].data['snapshot_idx'] - time_dist
            for edge in ['is t_cited by', 't_cites']:
                # print(graph.edges(etype=edge))
                # print(graph.edges[edge].data['w'])
                graph.edges[edge].data['w'] = edge_softmax(graph[edge], graph.edges[edge].data['w'])
                # print(graph.edges[edge].data['w'])
            # print(graph)
            # print(graph.nodes('snapshot'))
            # print(graph.nodes['paper'].data[dgl.NID])
        return graph

    def load_single_graph(self, graph, time_emb, emb_dim, graphs, start_time):
        a = graph.edges['is t_cited by'].data['w'].clone().detach()
        b = graph.edges['t_cites'].data['w'].clone().detach()
        graph.edges['is t_cited by'].data['w'] = b
        graph.edges['t_cites'].data['w'] = a
        # print(graph.edges(etype='is t_cited by'))
        # print(b)
        if self.max_time_length != self.time_length:
            graph = self.get_time_subgraph(graph)
        # print(graph)
        # print(graph.ntypes)
        for ntype in graph.ntypes:
            ndata = []
            snapshot_idx_list = graph.nodes[ntype].data['snapshot_idx'].numpy().tolist()
            cur_type = ntype
            if ntype == 'snapshot':
                cur_type = 'time'
                # nodes_list = (graph.nodes(ntype).numpy() + start_time).tolist()
                nodes_list = (graph.nodes[ntype].data['snapshot_idx'].numpy() + start_time).tolist()
                graph.nodes[ntype].data['time'] = (graph.nodes[ntype].data['snapshot_idx'] + start_time) \
                    .unsqueeze(dim=-1)
            else:
                nodes_list = graph.nodes[ntype].data[dgl.NID].numpy().tolist()
            for i in range(len(nodes_list)):
                node = nodes_list[i]
                if cur_type == 'time':
                    ndata.append(time_emb[start_time + snapshot_idx_list[i]][node].detach())
                else:
                    ndata.append(
                        graphs['all_graphs'][start_time + snapshot_idx_list[i]]
                            .nodes[cur_type].data['h'][node].detach())
            if ndata:
                graph.nodes[ntype].data['h'] = torch.stack(ndata, dim=0)
            else:
                graph.nodes[ntype].data['h'] = torch.zeros((0, emb_dim), dtype=torch.float32)
            cur_mask = []
            # for t in range(graphs['time_length']):
            for t in range(self.time_length):
                if (graph.nodes['paper'].data['snapshot_idx'] == t).sum() > 0:
                    cur_mask.append(1)
                else:
                    cur_mask.append(0)
            cur_mask = torch.tensor(cur_mask, dtype=torch.long)

        return graph, cur_mask

    def load_graphs(self, graphs):
        '''
        {
            'data': {phase: [batched_graphs, index_trans]},
            'node_trans': json.load(open(self.data_cat_path + 'sample_node_trans.json', 'r')),
            'time': {'train': train_time, 'val': val_time, 'test': test_time},
            'all_graphs': graphs,
        }
        '''
        time_emb = get_time_emb(graphs['all_graphs'])
        logger = logging.getLogger()
        # LOG = logging.getLogger(__name__)
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        # emb_dim = time_emb[list(time_emb.keys())[0]].shape[-1]
        emb_dim = graphs['all_graphs'][list(graphs['all_graphs'].keys())[0]].nodes['paper'].data['h'].shape[-1]
        for phase in graphs['data']:
            # print()
            # start_time = graphs['time'][phase] - (graphs['time_length'] - 1)
            start_time = graphs['time'][phase] - (self.time_length - 1)
            print(start_time)
            combined_graphs, trans_index = graphs['data'][phase]
            valid_masks = []
            dealt_graphs = []
            # emb load
            for graph_sample in tqdm(combined_graphs, file=tqdm_out, mininterval=30):
                if type(graph_sample) == list:
                    temp_list = []
                    for single_graph in graph_sample:
                        temp_graph, cur_mask = self.load_single_graph(single_graph, time_emb, emb_dim, graphs, start_time)
                        temp_list.append(temp_graph)
                        # print(temp_graph.nodes['snapshot'].data['h'])
                        # print(temp_graph.nodes['snapshot'].data['time'])
                    graph_sample = temp_list
                else:
                    graph_sample, cur_mask = self.load_single_graph(graph_sample, time_emb, emb_dim, graphs, start_time)
                    # print(cur_mask)
                valid_masks.append(cur_mask)
                # print(len(graph_sample))
                if self.max_time_length != self.time_length:
                    dealt_graphs.append(graph_sample)

            # trans_index = dict(zip(selected_papers, range(len(selected_papers))))
            # graphs['data'][phase][0] = combined_graphs
            if self.max_time_length != self.time_length:
                del combined_graphs
            else:
                dealt_graphs = combined_graphs
            # del combined_graphs
            graphs['data'][phase][0] = dealt_graphs
            graphs['data'][phase].append(valid_masks)
            # print(graphs['data'][phase][0])
            logging.info('-' * 30 + phase + ' done' + '-' * 30)
        # torch.save(graphs, './checkpoints/pubmed/c_sample_data')
        del graphs['all_graphs']
        # print(graphs.keys())
        return graphs

    def aug_graphs(self, graphs, aug_configs, aug_methods):
        # aug_methods, aug_configs = aug
        # all_graphs = augmenting_graphs_single(all_graphs, aug_config, aug_methods)
        logger = logging.getLogger()
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        logging.info('=' * 30 + 'augmenting start' + '=' * 30)
        if 'train' in graphs['data']:
            for i in tqdm(range(len(graphs['data']['train'][0])), file=tqdm_out, mininterval=30):
                g1, g2 = augmenting_graphs([graphs['data']['train'][0][i]], aug_configs, aug_methods)
                graphs['data']['train'][0][i] = g1 + g2
        return graphs

    def aug_graphs_plus(self, graphs, aug_configs, aug_methods):
        # aug_methods, aug_configs = aug
        # all_graphs = augmenting_graphs_single(all_graphs, aug_config, aug_methods)
        logger = logging.getLogger()
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        logging.info('=' * 30 + 'augmenting start' + '=' * 30)
        if 'train' in graphs['data']:
            for i in tqdm(range(len(graphs['data']['train'][0])), file=tqdm_out, mininterval=30):
                g1, g2 = augmenting_graphs_ctsgcn([graphs['data']['train'][0][i]], aug_configs, aug_methods)
                graphs['data']['train'][0][i] = g1 + g2
        return graphs

    def deal_graphs(self, graphs, data_path=None, path=None, log_path=None):
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
        :param graphs:
        :param data_path:
        :param path:
        :param log_path:
        :return:
        '''
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
        # dealt_graphs = {'data': {}, 'time': graphs['time'], 'node_trans': graphs['node_trans'],
        #                 'all_graphs': None}
        split_data = torch.load(data_path)
        # all_snapshots = {}
        all_trans_index = json.load(open(path + '_trans.json', 'r'))
        pub_times = list(graphs['all_graphs'].keys())
        # del graphs['all_graphs']
        del graphs['data']
        print('not get ego_subgraph for saving time')
        og_paper_oids = {}
        for pub_time in pub_times:
            og_paper_oids[pub_time] = graphs['all_graphs'][pub_time].nodes['paper'].data[dgl.NID].detach()
        del graphs['all_graphs']

        graphs['selected_papers'] = {}
        for phase in phases:
            graphs['selected_papers'][phase] = [graphs['node_trans']['paper'][paper] for paper in split_data[phase][0]]

        for phase in phases:
            # selected_papers = [graphs['node_trans']['paper'][paper] for paper in split_data[phase][0]]
            selected_papers = graphs['selected_papers'][phase]
            dealt_graph_list = []
            # print(range(graphs['time'][phase] - (graphs['time_length'] - 1), graphs['time'][phase] + 1))
            print(range(graphs['time'][phase] - (self.max_time_length - 1), graphs['time'][phase] + 1))
            # for pub_time in range(graphs['time'][phase] - (graphs['time_length'] - 1), graphs['time'][phase] + 1):
            for pub_time in range(graphs['time'][phase] - (self.max_time_length - 1), graphs['time'][phase] + 1):
                cur_graph_list = []
                start_time = time.time()
                cur_snapshot = joblib.load(path + '_' + str(pub_time) + '.job')
                logging.info('unbatching done! {:.2f}s'.format(time.time() - start_time))
                # cur_snapshot = torch.load(path + '_' + str(pub_time))
                for paper in selected_papers:
                    # cur_graph_list.append(all_snapshots[pub_time][all_trans_index[paper]])
                    cur_graph_list.append(cur_snapshot[all_trans_index[str(paper)]])
                # dealt_graph_list.append(dgl.batch(cur_graph_list))
                deal_graphs = get_time_graph(cur_graph_list, pub_time,
                                             # pub_time - graphs['time'][phase] + graphs['time_length'] - 1, tqdm_out)
                                             pub_time - graphs['time'][phase] + self.max_time_length - 1, tqdm_out)
                dealt_graph_list.append(deal_graphs)
            del cur_snapshot

            combined_graphs = []
            count = 0
            for snapshots in tqdm(zip(*dealt_graph_list), total=len(selected_papers),
                                  file=tqdm_out, mininterval=30):
                # links_dict = get_snapshot_links(snapshots, graphs['time'][phase] - (graphs['time_length'] - 1),
                #                                 og_paper_oids)
                links_dict = get_snapshot_links(snapshots, graphs['time'][phase] - (self.max_time_length - 1),
                                                og_paper_oids)
                # only valid snapshots
                invalid_count = 0
                for snapshot in snapshots:
                    if snapshot.nodes('paper').shape[0] == 0:
                        invalid_count += 1
                    else:
                        break
                # print(invalid_count)

                if invalid_count == len(snapshots):
                    print('bad data, invalid count', count)
                    print(snapshots)
                    raise Exception
                count += 1

                valid_snapshots = snapshots[invalid_count:]
                batched_snapshots = dgl.batch(valid_snapshots)
                # delete invalid nodes
                valid_links_dict = {}
                for edge in links_dict:
                    src_time, dst_time = edge[0] - invalid_count, edge[1] - invalid_count
                    if src_time < 0 | dst_time < 0:
                        print('bad data, time problem')
                        print(links_dict)
                        raise Exception
                    valid_links_dict[(src_time, dst_time)] = links_dict[edge]

                snapshots_cites_adj = get_snapshot_adj(valid_links_dict, len(valid_snapshots))
                # print(snapshots_cites_adj[0].shape)
                # print(snapshots_cites_adj[0].indices().numpy().tolist())
                src, dst = snapshots_cites_adj[0].indices().numpy().tolist()
                new_edges = {
                    ('snapshot', 't_cites', 'snapshot'): (src, dst),
                    ('snapshot', 'is t_cited by', 'snapshot'): (dst, src)
                }
                new_edata = {
                    't_cites': {'w': snapshots_cites_adj[0].values()},
                    'is t_cited by': {'w': snapshots_cites_adj[1].values()}
                }
                print(src, dst)
                print(snapshots_cites_adj[0].values())
                # print(new_edata['t_cites'].shape)
                # print(sum([snapshot.nodes('paper').shape[0] for snapshot in snapshots]))
                # print(batched_snapshots.nodes('paper').shape[0])
                combined_graph = add_new_elements(batched_snapshots, edges=new_edges, edata=new_edata)
                combined_graphs.append(combined_graph)

            trans_index = dict(zip(selected_papers, range(len(selected_papers))))
            dealt_graphs = [combined_graphs, trans_index]
            if path:
                joblib.dump(dealt_graphs, path.replace('TSGCN', 'CTSGCN') + '_' + phase + '.job')
            del dealt_graphs
            logging.info('-' * 30 + phase + ' done' + '-' * 30)

        if path:
            torch.save([], path.replace('TSGCN', 'CTSGCN'))
        return {}

    def get_snapshot_type(self, pub_distance):
        # 0 new, 1 normal, 2 old
        snapshot_type = torch.ones_like(pub_distance)
        new_papers = torch.where(pub_distance < self.cut_distance)
        old_papers = torch.where(pub_distance > 2 * self.cut_distance - 1)
        snapshot_type[new_papers] = 0
        snapshot_type[old_papers] = 2
        # print(snapshot_type)
        return snapshot_type

    def get_single_indicator(self, combined_graph, pub_time, target_id, core_id, mask, batch_index):
        if type(combined_graph) == list:
            for single_graph in combined_graph:
                pub_distance = single_graph.nodes['snapshot'].data['time'] - pub_time
                single_graph.nodes['snapshot'].data['type'] = self.get_snapshot_type(
                    pub_distance.squeeze(dim=-1))
                single_graph.nodes['snapshot'].data['batch_idx'] = torch.zeros(pub_distance.shape[0], dtype=torch.long) \
                                                                   + batch_index
                # print(single_graph.nodes['snapshot'].data['time'].shape)
                if mask:
                    single_graph.nodes['snapshot'].data['mask'] = torch.zeros_like(
                        single_graph.nodes['snapshot'].data['type'], dtype=torch.long)
                else:
                    single_graph.nodes['snapshot'].data['mask'] = torch.ones_like(
                        single_graph.nodes['snapshot'].data['type'], dtype=torch.long)
                for ntype in single_graph.ntypes:
                    node_count = single_graph.nodes(ntype).shape[0]
                    single_graph.nodes[ntype].data['target'] = torch.zeros(node_count, dtype=torch.long) + target_id
                    single_graph.nodes[ntype].data['core'] = torch.zeros(node_count, dtype=torch.long) + core_id
        else:
            pub_distance = combined_graph.nodes['snapshot'].data['time'] - pub_time
            combined_graph.nodes['snapshot'].data['type'] = self.get_snapshot_type(pub_distance.squeeze(dim=-1))
            combined_graph.nodes['snapshot'].data['batch_idx'] = torch.zeros(pub_distance.shape[0], dtype=torch.long) \
                                                                 + batch_index
            if mask:
                combined_graph.nodes['snapshot'].data['mask'] = torch.zeros_like(
                    combined_graph.nodes['snapshot'].data['type'], dtype=torch.long)
            else:
                combined_graph.nodes['snapshot'].data['mask'] = torch.ones_like(
                    combined_graph.nodes['snapshot'].data['type'], dtype=torch.long)
            for ntype in combined_graph.ntypes:
                node_count = combined_graph.nodes(ntype).shape[0]
                combined_graph.nodes[ntype].data['target'] = torch.zeros(node_count, dtype=torch.long) + target_id
                combined_graph.nodes[ntype].data['core'] = torch.zeros(node_count, dtype=torch.long) + core_id

    def get_indicators(self, all_graphs, ids, times):
        # print(len(all_graphs))
        if self.hn and self.training:
            # if self.hn:
            true_len = len(all_graphs) // self.hn
            # print('indicator len:', true_len)
            # print(true_len, len(set(ids.numpy().tolist())))
            for i in range(true_len):
                # print(ids[i])
                combined_graph = all_graphs[i]
                self.get_single_indicator(combined_graph, times[i], ids[i], ids[i], False, i)

                for j in range(self.hn - 1):
                    cur_index = i + true_len * (j + 1)
                    combined_graph = all_graphs[cur_index]
                    self.get_single_indicator(combined_graph, times[cur_index], ids[i], ids[cur_index], False, cur_index)

        else:
            for i in range(len(all_graphs)):
                combined_graph = all_graphs[i]
                self.get_single_indicator(combined_graph, times[i], ids[i], ids[i], False, i)

        return all_graphs

    def get_hn_graphs(self, trans_index, combined_graphs_list, ids, df_hn, raw_embs=None):
        new_ids = []
        new_times = []
        new_embs = []
        all_papers = list(trans_index.keys())
        raw_papers = set(ids.numpy().tolist())
        if self.hn_method == 'co_ref':
            columns = ['co_ref', 'co_cite', 'in_ref', 'in_cite']
        elif self.hn_method == 'co_cite':
            columns = ['co_cite', 'co_ref', 'in_cite', 'in_ref']
        else:
            columns = ['co_ref', 'co_cite', 'in_ref', 'in_cite']
        # print('hn_columns:', columns)
        # self.hn = 2
        all_graphs = [combined_graphs_list[trans_index[paper.item()]] for paper in ids]
        # print(df_hn.head())
        df_time = pd.DataFrame(df_hn['pub_time'], index=df_hn.index)
        df_hn = df_hn.loc[ids.numpy()]

        target_dict = {paper.item(): {paper.item()} for paper in ids}
        for paper in ids:
            # cur_exist_set = set(target_dict[paper.item()])
            cur_comp = df_hn.loc[paper.item()]
            comp_paper = None
            for column in columns:
                # target_dict[paper.item()]
                cur_list = [comp_paper for comp_paper in cur_comp[column] if comp_paper not in target_dict[paper.item()]]
                if len(cur_list) <= (self.hn + 1 - len(target_dict[paper.item()])):
                    target_dict[paper.item()].update(set(cur_list))
                else:
                    for i in range(self.hn + 1 - len(target_dict[paper.item()])):
                        while (comp_paper is None) | (comp_paper in target_dict[paper.item()]):
                            comp_paper = cur_list[np.random.randint(len(cur_list))]
                        target_dict[paper.item()].add(comp_paper)
            for n in range(self.hn + 1 - len(target_dict[paper.item()])):
                if 'label' in self.cl_type:
                    # print('here')
                    cur_label = cur_comp['label']
                    cur_papers = self.re_group[cur_label]
                    while (comp_paper is None) | (comp_paper in target_dict[paper.item()]):
                        comp_paper = cur_papers[np.random.randint(len(cur_papers))]
                else:
                    while (comp_paper is None) | (comp_paper in target_dict[paper.item()]):
                        comp_paper = all_papers[np.random.randint(len(all_papers))]
                target_dict[paper.item()].add(comp_paper)
            target_dict[paper.item()].discard(paper.item())
            target_dict[paper.item()] = list(target_dict[paper.item()])
            # print(paper.item(), target_dict[paper.item()])
            if len(target_dict[paper.item()]) != self.hn:
                print('what is wrong')
                raise Exception
        # print('get_paper_time: {:.3f}s'.format(time.time() - start_time))

        # start_time = time.time()
        # raw papers decide whether to copy or just use
        for i in range(self.hn - 1):
            # print(comp_paper)
            # a new graph for learning
            # all_graphs.append(copy.deepcopy(combined_graphs_list[trans_index[comp_paper]]))3
            for paper in ids:
                comp_paper = target_dict[paper.item()][i]
                # all_graphs.append([add_new_elements(g) for g in combined_graphs_list[trans_index[comp_paper]]])
                # all_graphs.append([copy.deepcopy(g) for g in combined_graphs_list[trans_index[comp_paper]]])
                all_graphs.append([copy.deepcopy(g) if comp_paper in raw_papers else g
                                   for g in combined_graphs_list[trans_index[comp_paper]]])
                raw_papers.add(comp_paper)
                new_ids.append(comp_paper)
                new_times.append(int(df_time.at[comp_paper, 'pub_time']))
                if raw_embs is not None:
                    new_embs.append(raw_embs[trans_index[comp_paper]])
        # print('get_copy_time: {:.3f}s'.format(time.time() - start_time))
        # print(len(new_ids))
        return all_graphs, new_ids, new_times, new_embs

    def get_graphs(self, content, lengths, masks, ids, graph, times, phase='eval', df_hn=None):
        combined_graphs_list, trans_index, valid_masks = graph
        # print(len(combined_graphs_list))
        # graphs_list = dgl.unbatch(batched_graphs)
        new_ids = []
        new_times = []
        # start_time = time.time()
        # print(len(combined_graphs_list[0]))
        if df_hn is not None:
            all_graphs, new_ids, new_times, _ = self.get_hn_graphs(trans_index, combined_graphs_list, ids, df_hn)
        else:
            # self.hn = False
            all_graphs = [combined_graphs_list[trans_index[paper.item()]] for paper in ids]
        if len(new_ids) > 0:
            new_ids = torch.tensor(new_ids, dtype=torch.long)
            ids = torch.cat((ids, new_ids), dim=0)
            new_times = torch.tensor(new_times, dtype=torch.long)
            times = torch.cat((times, new_times), dim=0)
        all_graphs = self.get_indicators(all_graphs, ids, times)
        # print('get_indicator_time: {:.3f}s'.format(time.time() - start_time))

        cur_masks = torch.stack([valid_masks[trans_index[paper.item()]] for paper in ids], dim=0).to(self.device)

        # start_time = time.time()
        if phase == 'train':
            g1_list = dgl.batch([combined_graph[0] for combined_graph in all_graphs]).to(self.device)
            g2_list = dgl.batch([combined_graph[1] for combined_graph in all_graphs]).to(self.device)
            # return g1_list, g2_list
            return [g1_list, ids, cur_masks], [g2_list, ids, cur_masks]
        elif phase == 'train_combined':
            g1_list = [combined_graph[0] for combined_graph in all_graphs]
            g2_list = [combined_graph[1] for combined_graph in all_graphs]
            all_graphs = dgl.batch(g1_list + g2_list).to(self.device)
            # print('batch_time: {:.3f}s'.format(time.time() - start_time))
            return [all_graphs, ids, cur_masks]
        else:
            all_graphs = dgl.batch([combined_graph for combined_graph in all_graphs]).to(self.device)
            # return all_graphs
            return [all_graphs, ids, cur_masks]

    def encode(self, inputs, get_attention=False):
        # all_graphs = inputs
        all_graphs, paper_ids, valid_masks = inputs
        if self.ntype_fc:
            all_graphs.srcdata['h'] = self.ntype_fc(all_graphs.srcdata['h'])
        # print(all_graphs.nodes['paper'].data.keys())

        result_srcdata = {}
        snapshot_emb = self.snapshot_embedding(all_graphs.srcdata['snapshot_idx'])
        for ntype in self.ntypes:
            # print(all_graphs.srcdata['h'].keys())
            result_srcdata[ntype] = all_graphs.srcdata['h'][ntype] + snapshot_emb[ntype]
        all_graphs.srcdata['h'] = result_srcdata
        # print('before gcn', [torch.isnan(result_srcdata[e]).sum() for e in result_srcdata])

        if get_attention:
            out_emb, all_list, attn_list = self.graph_encoder(all_graphs, all_graphs.srcdata['h'], get_attention=True)
        else:
            out_emb, all_list = self.graph_encoder(all_graphs, all_graphs.srcdata['h'])
        # if self.gcn_out == 'mean':
        #     out_emb[self.pred_type] = torch.stack([ndata[self.pred_type] for ndata in all_list], dim=0).mean(dim=0)
        # elif self.gcn_out == 'concat':
        #     out_emb[self.pred_type] = self.fc_out(torch.cat([ndata[self.pred_type] for ndata in all_list], dim=-1))
        if self.gcn_out == 'mean':
            for ntype in self.ntypes:
                out_emb[ntype] = torch.stack([ndata[ntype] for ndata in all_list], dim=0).mean(dim=0)
                # print(out_emb[ntype].shape)
        elif self.gcn_out == 'concat':
            for ntype in self.ntypes:
                out_emb[ntype] = self.fc_out[ntype](torch.cat([ndata[ntype] for ndata in all_list], dim=-1))

        all_graphs.dstdata['h'] = out_emb

        if self.pred_type == 'paper':
            # readout = dgl.readout_nodes(all_graphs, 'h', op=self.time_pooling, ntype='paper')
            # print(all_graphs.nodes['paper'].data['h'].shape, all_graphs.nodes['paper'].data['w'].shape)
            readout = dgl.readout_nodes(all_graphs, 'h', 'w', ntype='paper')
        elif self.pred_type == 'snapshot':
            readout = dgl.readout_nodes(all_graphs, 'h', op=self.time_pooling, ntype='snapshot')
        # print(time_out.shape)
        # print('readout', torch.isnan(readout).sum())
        # print('readout', readout)
        # if self.hn and self.training:
        #     true_len = readout.shape[0] // self.hn
        #     readout = readout[:true_len]

        if get_attention:
            return readout, all_graphs, attn_list
        else:
            return readout, all_graphs

    def decode(self, time_out):
        # if self.hn:
        #     true_len = time_out.shape[0] // 2
        #     output = self.fc(time_out[:true_len])
        # else:
        #     output = self.fc(time_out)
        #     return output
        # output = self.fc(time_out)
        if self.hn and self.training:
            true_len = time_out.shape[0] // self.hn
            time_out = time_out[:true_len]
        # print(time_out.shape)
        if self.norm:
            time_out = self.norm(time_out)
        if self.dn:
            time_out = self.dn(time_out)

        output_reg = self.fc_reg(time_out)
        output_cls = self.fc_cls(time_out)
        return output_reg, output_cls

    def forward(self, content, lengths, masks, ids, graph, times, **kwargs):
        inputs = self.get_graphs(content, lengths, masks, ids, graph, times)
        time_out, _ = self.encode(inputs)
        output = self.decode(time_out)
        # output = torch.relu(output)
        # print(output)
        return output

    def get_information_graph(self, inputs, attn_list):
        # rgat + cgin
        # attn_list = list(zip(*attn_list))
        # print(len(attn_list[0]), len(attn_list[1]))
        # print('rgat + cgin')
        all_graphs, ids, cur_masks = inputs
        for ntype in all_graphs.ntypes:
            all_graphs.nodes[ntype].data.pop('h')
        for k in range(len(attn_list)):
            all_graphs.edges[('paper', 'is in', 'snapshot')].data['a_{}'.format(k)] = attn_list[k][0]
            all_graphs.edges[('paper', 'cites', 'paper')].data['a_{}'.format(k)] = attn_list[k][1]
            # print(attn_list[k][1].shape)
        all_graphs = dgl.unbatch(all_graphs)
        dealt_graphs = []
        for graph in all_graphs:
            # dealt_graphs.append(dgl.node_type_subgraph(graph, ntypes=['paper', 'snapshot']))
            dealt_graphs.append(dgl.edge_type_subgraph(graph, etypes=[('paper', 'is in', 'snapshot'),
                                                                      ('paper', 'cites', 'paper')]))
        # print(len(dealt_graphs))
        return dealt_graphs

    def get_information_weight(self, inputs, attn_list):
        '''
        get the attention weight of C-GIN and R-GAT
        :param inputs:
        :param attn_list:
        :return:
        '''
        # rgat + cgin
        # attn_list = list(zip(*attn_list))
        # print(len(attn_list[0]), len(attn_list[1]))
        # print('rgat + cgin')
        all_graphs, ids, cur_masks = inputs
        # for ntype in all_graphs.ntypes:
        #     all_graphs.nodes[ntype].data.pop('h')
        for k in range(len(attn_list)):
            all_graphs.edges[('paper', 'is in', 'snapshot')].data['a_{}'.format(k)] = attn_list[k][0]
            all_graphs.edges[('paper', 'cites', 'paper')].data['a_{}'.format(k)] = attn_list[k][1]
        all_graphs = dgl.unbatch(all_graphs)
        # dealt_graphs = []
        # for graph in all_graphs:
        #     # dealt_graphs.append(dgl.node_type_subgraph(graph, ntypes=['paper', 'snapshot']))
        #     dealt_graphs.append(dgl.edge_type_subgraph(graph, etypes=[('paper', 'is in', 'snapshot'),
        #                                                               ('paper', 'cites', 'paper')]))
        # print(len(dealt_graphs))
        all_cgin_result = []
        all_rgat_result = []
        for cur_graph in all_graphs:
            # CGIN weight
            edata = cur_graph.edges[('paper', 'cites', 'paper')].data
            # print(edata.keys())
            temp_list = []
            for key in [key for key in edata.keys() if key.startswith('a_')]:
                temp_list.append(edata[key].cpu().numpy())
            cgin_weight = np.mean(np.array(temp_list), axis=0)
            src_index = cur_graph.edges(etype=('paper', 'cites', 'paper'))[0]
            # src_citation = cur_graph.nodes['paper'].data['citations'][src_index].unsqueeze(dim=-1).cpu().numpy()
            src_citation = cur_graph.nodes['paper'].data['citations'][src_index]
            src_citation_raw = src_citation.unsqueeze(dim=-1).cpu().numpy()
            # print(src_citation)
            src_citation = edge_softmax(cur_graph['paper', 'cites', 'paper'],
                                        src_citation.float()).unsqueeze(dim=-1).cpu().numpy()
            # print(src_citation)
            cgin_out = np.concatenate((cgin_weight, src_citation, src_citation_raw), axis=-1)
            all_cgin_result.append(cgin_out)

            # RGAT weight
            count_list = []
            result_list = []
            for t in range(self.time_length):
                temp_list = []
                cur_snapshot = cur_graph.nodes('snapshot')[torch.where(cur_graph.nodes['snapshot'].data['snapshot_idx'] == t)]
                if cur_snapshot.shape[0] > 0:
                    cur_index = torch.where(cur_graph.nodes['paper'].data['snapshot_idx'] == t)
                    cur_subgraph = dgl.node_subgraph(cur_graph, nodes={
                        'paper': cur_graph.nodes('paper')[cur_index],
                        'snapshot': cur_snapshot,
                    }, store_ids=False)
                    # edges = cur_subgraph.edges[('paper', 'is in', 'snapshot')]
                    # print(cur_subgraph)
                    # print(cur_subgraph.edges[('paper', 'is in', 'snapshot')].data.keys())
                    # print(cur_subgraph.edges[('paper', 'is in', 'snapshot')].data['a_0'])
                    assert torch.equal(cur_subgraph.nodes('paper'),
                                       cur_subgraph.edges(etype=('paper', 'is in', 'snapshot'))[0])
                    # print()
                    edata = cur_subgraph.edges[('paper', 'is in', 'snapshot')].data
                    count_list.append([cur_subgraph.nodes['paper'].data['is_ref'].sum().item(),
                                       cur_subgraph.nodes['paper'].data['is_cite'].sum().item(),
                                       cur_subgraph.nodes['paper'].data['is_target'].sum().item()])
                    for key in edata.keys():
                        # print('=' * 59)
                        ref_weight = (edata[key].squeeze(dim=-1) * cur_subgraph.nodes['paper'].data['is_ref'].unsqueeze(
                            dim=-1)).sum(dim=0).mean().item()
                        cite_weight = (edata[key].squeeze(dim=-1) * cur_subgraph.nodes['paper'].data['is_cite'].unsqueeze(
                            dim=-1)).sum(dim=0).mean().item()
                        target_weight = (
                                    edata[key].squeeze(dim=-1) * cur_subgraph.nodes['paper'].data['is_target'].unsqueeze(
                                dim=-1)).sum(dim=0).mean().item()
                        temp_list.append([ref_weight, cite_weight, target_weight])
                    temp_list = np.mean(np.array(temp_list), axis=0)
                    # print(temp_list.shape)
                    result_list.append(temp_list)
                else:
                    count_list.append(np.zeros(3, dtype=np.float32))
                    result_list.append(np.zeros(3, dtype=np.float32))
            count_list = np.array(count_list)
            result_list = np.array(result_list)
            all_array = np.concatenate((count_list, result_list), axis=-1)
            # print(cgin_out.shape)
            # print(all_array.shape)
            all_rgat_result.append(all_array)

        return all_cgin_result, np.array(all_rgat_result)


    def show(self, content, lengths, masks, ids, graph, times, **kwargs):
        inputs = self.get_graphs(content, lengths, masks, ids, graph, times)
        if kwargs.get('return_graph', False):
            time_out, _, attn_list = self.encode(inputs, get_attention=True)
            dealt_graphs = self.get_information_graph(inputs, attn_list)
        # dealt_graphs = []
            return [time_out.cpu().numpy(), dealt_graphs]
        elif kwargs.get('return_weight', False):
            time_out, _, attn_list = self.encode(inputs, get_attention=True)
            cgin, rgat = self.get_information_weight(inputs, attn_list)
            return [time_out.cpu().numpy(), cgin, rgat]
        else:
            time_out, _ = self.encode(inputs)
            output = self.decode(time_out)
            return [time_out.cpu().numpy()]


def get_time_emb(all_graphs_dict, time_list=None):
    # 改版的方法，直接读取所有年的time_emb再赋值
    time_emb = {snapshot_time: {} for snapshot_time in all_graphs_dict}
    for snapshot_time in all_graphs_dict:
        cur_time_embs = all_graphs_dict[snapshot_time].nodes['time'].data['h']
        time_oids = all_graphs_dict[snapshot_time].nodes['time'].data[dgl.NID].numpy().tolist()
        # print(time_oids)
        time_oids_cid_trans = dict(zip(time_oids, range(len(time_oids))))
        if time_list is None:
            cur_time_list = time_oids
        else:
            cur_time_list = time_list
        for pub_time in cur_time_list:
            time_emb[snapshot_time][pub_time] = cur_time_embs[time_oids_cid_trans[pub_time]]
        # print(time_emb[snapshot_time].keys())

    return time_emb


def get_snapshot_links(graph_list, start_time, og_paper_oids):
    all_times = []
    reversed_time_trans = {}
    # print(graph_list)
    for i in range(len(graph_list)):
        graph = graph_list[i]
        # og = all_graphs[start_time + i]
        # print(og)
        # print(graph)
        graph.nodes['paper'].data['oid'] = og_paper_oids[start_time + i][graph.nodes['paper'].data[dgl.NID]]
        oid_trans = og_paper_oids[start_time + i][graph.nodes['paper'].data[dgl.NID]].tolist()
        reversed_time_trans[i] = dict(zip(oid_trans, range(len(oid_trans))))
        all_times.append(oid_trans)

    # combined all snapshots with time links
    time_links = defaultdict(int)
    time_nodes = {}
    time_nodes[0] = set(all_times[0])
    all_times_set = [set(item) for item in all_times]
    for i in range(1, len(all_times)):
        time_nodes[i] = all_times_set[i] - all_times_set[i - 1]
    # print(time_nodes)

    inverted_time_nodes = {}
    for time_idx in time_nodes:
        for node in time_nodes[time_idx]:
            inverted_time_nodes[node] = time_idx
    # print(inverted_time_nodes)

    for i in range(1, len(all_times)):
        for node in time_nodes[i]:
            cites_papers = graph_list[i].out_edges(reversed_time_trans[i][node], etype='cites')[1].numpy().tolist()
            for cites_paper in cites_papers:
                time_links[(i, inverted_time_nodes[all_times[i][cites_paper]])] += 1

    # print(time_links)
    # print(np.array(list(time_links.keys())))
    # print(torch.tensor(list(time_links.keys())))
    # print(time_links.keys())
    if len(time_links.keys()) == 0:
        for time_idx, nodes in time_nodes.items():
            if len(nodes) > 0:
                time_links[(time_idx, time_idx)] = 1

    return time_links


def get_snapshot_adj(time_links, time_length):
    src, dst = zip(*list(time_links.keys()))
    time_cites_adj = torch.sparse_coo_tensor(indices=torch.tensor(list(zip(src, dst))).T,
                                             values=torch.tensor(list(time_links.values())),
                                             size=(time_length, time_length), dtype=torch.float32)
    time_cited_adj = torch.sparse_coo_tensor(indices=torch.tensor(list(zip(dst, src))).T,
                                             values=torch.tensor(list(time_links.values())),
                                             size=(time_length, time_length), dtype=torch.float32)
    time_cites_adj = get_norm_adj(time_cites_adj)
    time_cited_adj = get_norm_adj(time_cited_adj)
    # print(time_cites_adj.to_dense())
    return [time_cites_adj, time_cited_adj]


def get_time_graph(unbatched_graphs, t, i, tqdm_out=None, cur_emb_graph=None, time_emb=None):
    new_graphs_list = []
    for graph in tqdm(unbatched_graphs, file=tqdm_out, mininterval=30):
        # print(graph.ntypes)
        if cur_emb_graph:
            for ntype in graph.ntypes:
                cur_index = graph.nodes[ntype].data[dgl.NID]
                graph.nodes[ntype].data['h'] = cur_emb_graph.nodes[ntype].data['h'][cur_index].detach()
        time_dict = IndexDict()
        cur_papers = graph.nodes('paper').numpy().tolist()
        paper_time = graph.nodes['paper'].data['time'].squeeze(dim=-1).numpy().tolist()
        # cur_times = sorted(set(paper_time))

        trans_times = [time_dict[pub_time] for pub_time in paper_time]
        new_edges = {
            ('paper', 'is in', 'snapshot'): (cur_papers, [0] * len(cur_papers)),
            ('snapshot', 'has', 'paper'): ([0] * len(cur_papers), cur_papers),
            ('paper', 'is shown in', 'time'): (cur_papers, trans_times),
            ('time', 'shows', 'paper'): (trans_times, cur_papers)
        }
        # if len(cur_papers) > 0:
        #     snapshot_count = 1
        # else:
        #     snapshot_count = 0
        if time_emb:
            ndata = {'snapshot': {'h': time_emb[t].detach().unsqueeze(dim=0)}}
            if time_dict:
                ndata['time'] = {
                    'h': torch.stack([time_emb[pub_time].detach() for pub_time in time_dict.keys()], dim=0),
                    dgl.NID: torch.tensor(list(time_dict.keys()), dtype=torch.long)}
            else:
                ndata['time'] = {'h': torch.zeros(0, time_emb[t].shape[-1], dtype=torch.float32),
                                 dgl.NID: torch.zeros(0, dtype=torch.long)}
            cur_new_graph = add_new_elements(graph, {'snapshot': 1, 'time': len(time_dict)},
                                             ndata, new_edges, {})
        else:
            if time_dict:
                ndata = {'time': {dgl.NID: torch.tensor(list(time_dict.keys()), dtype=torch.long)}}
            else:
                ndata = {'time': {dgl.NID: torch.zeros(0, dtype=torch.long)}}
            cur_new_graph = add_new_elements(graph, {'snapshot': 1, 'time': len(time_dict)},
                                             ndata, new_edges, {})

        for ntype in cur_new_graph.ntypes:
            cur_new_graph.nodes[ntype].data['snapshot_idx'] = \
                torch.tensor([i] * cur_new_graph.nodes(ntype).shape[0],
                             dtype=torch.long)
        new_graphs_list.append(cur_new_graph)
    return new_graphs_list


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    etypes = ['cites', 'is cited by', 'writes', 'is writen by', 'publishes', 'is published by',
              'shows', 'is shown in', 'is in', 'has', "t_cites", "is t_cited by"]
    ntypes = ['paper', 'author', 'journal', 'time', 'snapshot']

    # data = torch.load('../checkpoints/pubmed/c_sample_data')
    # # print()
    # trans_dict = data['data']['train'][1]
    # random.seed(123)
    # papers = list(trans_dict.keys())
    # # random.shuffle()
    # sample_ids = [4724, 12968, 8536, 9558]
    #
    # cur_paper = sample_ids[3]
    # print(trans_dict[cur_paper])
    # print(len(data['data']['train'][0]))
    # # print(data['data']['train'][0][-1][trans_dict[cur_paper]])
    # # print(data['data']['train'][2][trans_dict[cur_paper]])
    # print(data['data']['train'][0][trans_dict[cur_paper]])
    # print(data['data']['train'][0][trans_dict[cur_paper]].nodes['paper'].data[dgl.NID])
    #
    model = CTSGCN(0, 300, 1, 0, word2vec=None, dropout=0.5, n_layers=3,
                   pred_type='paper', time_pooling='mean', hop=2,
                   time_length=10, hidden_dim=256, out_dim=128, ntypes=ntypes, etypes=etypes)
    for child_name, child in model.named_children():
        print(child_name)
        print(child)
    # print(model(None, None, None, torch.tensor(sample_ids), data['data']['train']))
    # for name, param in model.graph_encoder.named_parameters():
    #     print(name)
    #     print(param.shape)

    # sample_dict = {}
    # for t in range(2005, 2010):
    #     sample_dict[t] = torch.load('../data/pubmed/graph_sample_feature_vector_{}'.format(t))
    # time_emb = get_time_emb(sample_dict)
    # print(sample_dict[2005])
    # print(time_emb[2005].keys())
    # print(time_emb[2005][2005].shape)
    print(model.state_dict().keys())
    state_dict = model.state_dict()
    for key in state_dict:
        if key.startswith('fc'):
            print(key, state_dict[key])
