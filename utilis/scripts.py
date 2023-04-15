import json
import math
import random

import dgl
import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score, \
    precision_recall_curve, auc
import torch.nn.functional as F


def eval_result(all_true_values, all_predicted_values):
    # print(mean_absolute_error(all_true_values, all_predicted_values, multioutput='raw_values'))
    all_true_values = all_true_values.flatten(order='C')
    # print(all_true_values)
    all_predicted_values = all_predicted_values.flatten(order='C')
    # 控制不超出上界
    all_predicted_values = np.where(all_predicted_values == np.inf, 1e20, all_predicted_values)
    # print(all_predicted_values)
    mae = mean_absolute_error(all_true_values, all_predicted_values)
    mse = mean_squared_error(all_true_values, all_predicted_values)
    rmse = math.sqrt(mse)
    r2 = r2_score(all_true_values, all_predicted_values)

    rse = (all_true_values - all_predicted_values) ** 2 / (all_true_values + 1)
    Mrse = np.mean(rse, axis=0)
    mrse = np.percentile(rse, 50)

    acc = np.mean(np.all(((0.5 * all_true_values) <= all_predicted_values,
                  all_predicted_values <= (1.5 * all_true_values)), axis=0), axis=0)

    # print(all_true_values)
    # add min for log
    log_true_values = np.log(1 + all_true_values)
    log_predicted_values = np.log(1 + np.where(all_predicted_values >= 0, all_predicted_values, 0))
    male = mean_absolute_error(log_true_values, log_predicted_values)
    msle = mean_squared_error(log_true_values, log_predicted_values)
    rmsle = math.sqrt(msle)
    log_r2 = r2_score(log_true_values, log_predicted_values)
    smape = np.mean(np.abs(log_true_values - log_predicted_values) /
                    ((np.abs(log_true_values) + np.abs(log_predicted_values) + 0.1) / 2), axis=0)
    mape = np.mean(np.abs(log_true_values - log_predicted_values) / (np.abs(log_true_values) + 0.1), axis=0)
    # print('true', all_true_values[:10])
    # print('pred', all_predicted_values[:10])
    return [acc, mae, r2, mse, rmse, Mrse, mrse, male, log_r2, msle, rmsle, smape, mape]


def eval_aux_results(all_true_label, all_predicted_result):
    all_predicted_label = np.argmax(all_predicted_result, axis=1)
    # all
    acc = accuracy_score(all_true_label, all_predicted_label)
    one_hot = np.eye(all_predicted_result.shape[-1])[all_true_label]
    try:
        roc_auc = roc_auc_score(one_hot, all_predicted_result)
        log_loss_value = log_loss(all_true_label, all_predicted_result)
    except Exception as e:
        print(e)
        roc_auc = 0
        log_loss_value = 0
    # print(all_predicted_result)
    # # pos
    # prec = precision_score(all_true_label, all_predicted_label)
    # recall = recall_score(all_true_label, all_predicted_label)
    # f1 = f1_score(all_true_label, all_predicted_label, average='binary')
    # pr_p, pr_r, _ = precision_recall_curve(all_true_label, all_predicted_result[:, 1])
    # pr_auc = auc(pr_r, pr_p)
    # # neg
    # prec_neg = precision_score(all_true_label, all_predicted_label, pos_label=0)
    # recall_neg = recall_score(all_true_label, all_predicted_label, pos_label=0)
    # f1_neg = f1_score(all_true_label, all_predicted_label, average='binary', pos_label=0)
    # pr_np, pr_nr, _ = precision_recall_curve(all_true_label, all_predicted_result[:, 0], pos_label=0)
    # pr_auc_neg = auc(pr_nr, pr_np)
    # for all
    micro_prec = precision_score(all_true_label, all_predicted_label, average='micro')
    micro_recall = recall_score(all_true_label, all_predicted_label, average='micro')
    micro_f1 = f1_score(all_true_label, all_predicted_label, average='micro')
    macro_prec = precision_score(all_true_label, all_predicted_label, average='macro')
    macro_recall = recall_score(all_true_label, all_predicted_label, average='macro')
    macro_f1 = f1_score(all_true_label, all_predicted_label, average='macro')

    results = [acc, roc_auc, log_loss_value, micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1]
    return results


def eval_aux_labels(all_true_label, all_predicted_label):
    # acc, mi_prec, mi_recall, mi_f1, ma_prec, ma_recall, ma_f1
    # all_predicted_label = np.argmax(all_predicted_result, axis=1)
    # all
    acc = accuracy_score(all_true_label, all_predicted_label)
    # one_hot = np.eye(all_predicted_result.shape[-1])[all_true_label]
    # try:
    #     roc_auc = roc_auc_score(one_hot, all_predicted_result)
    #     log_loss_value = log_loss(all_true_label, all_predicted_result)
    # except Exception as e:
    #     print(e)
    #     roc_auc = 0
    #     log_loss_value = 0
    # print(all_predicted_result)
    # # pos
    # prec = precision_score(all_true_label, all_predicted_label)
    # recall = recall_score(all_true_label, all_predicted_label)
    # f1 = f1_score(all_true_label, all_predicted_label, average='binary')
    # pr_p, pr_r, _ = precision_recall_curve(all_true_label, all_predicted_result[:, 1])
    # pr_auc = auc(pr_r, pr_p)
    # # neg
    # prec_neg = precision_score(all_true_label, all_predicted_label, pos_label=0)
    # recall_neg = recall_score(all_true_label, all_predicted_label, pos_label=0)
    # f1_neg = f1_score(all_true_label, all_predicted_label, average='binary', pos_label=0)
    # pr_np, pr_nr, _ = precision_recall_curve(all_true_label, all_predicted_result[:, 0], pos_label=0)
    # pr_auc_neg = auc(pr_nr, pr_np)
    # for all
    micro_prec = precision_score(all_true_label, all_predicted_label, average='micro')
    micro_recall = recall_score(all_true_label, all_predicted_label, average='micro')
    micro_f1 = f1_score(all_true_label, all_predicted_label, average='micro')
    macro_prec = precision_score(all_true_label, all_predicted_label, average='macro')
    macro_recall = recall_score(all_true_label, all_predicted_label, average='macro')
    macro_f1 = f1_score(all_true_label, all_predicted_label, average='macro')

    results = [acc, micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1]
    return results


def result_format(results):
    loss, acc, mae, r2, mse, rmse, Mrse, mrse, male, log_r2, msle, rmsle, smape, mape = results
    format_str = '| loss {:8.4f} | acc {:9.4f} |\n' \
                 '| MAE {:9.4f} | R2 {:10.4f} |\n' \
                 '| MSE {:9.4f} | RMSE {:8.4f} |\n'\
                 '| MRSE {:8.4f} | mRSE {:8.4f} |\n'\
                 '| MALE {:8.4f} | LR2 {:9.4f} |\n'\
                 '| MSLE {:8.4f} | RMSLE {:7.4f} |\n'\
                 '| SMAPE {:8.4f} | MAPE {:7.4f} |\n'.format(loss, acc,
                                                       mae, r2,
                                                       mse, rmse,
                                                       Mrse, mrse,
                                                       male, log_r2,
                                                       msle, rmsle,
                                                       smape, mape) + \
                 '-' * 59
    print(format_str)
    return format_str

def aux_result_format(results, phase):
    avg_loss, acc, roc_auc, log_loss_value, micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1 = results

    format_str = '| aux loss {:8.3f} | {:9} {:7.3f} |\n' \
                 '| roc_auc {:9.3f} | log_loss {:8.3f} |\n' \
                 'micro:\n' \
                 '| precision {:7.3f} | recall {:10.3f} |\n' \
                 '| f1 {:14.3f} |\n' \
                 'macro:\n' \
                 '| precision {:7.3f} | recall {:10.3f} |\n' \
                 '| f1 {:14.3f} |\n'.format(avg_loss, phase + ' acc', acc, roc_auc, log_loss_value,
                                            micro_prec, micro_recall, micro_f1,
                                            macro_prec, macro_recall, macro_f1)
    print(format_str)
    return format_str


def get_configs(data_source, model_list, replace_dict=None):
    fr = open('./configs/{}.json'.format(data_source))
    configs = json.load(fr)
    full_configs = {'default': configs['default']}
    if replace_dict:
        print('>>>>>>>>>>replacing here<<<<<<<<<<<<<<<')
        print(replace_dict)
        for key in replace_dict:
            full_configs['default'][key] = replace_dict[key]
    for model in model_list:
        full_configs[model] = configs['default'].copy()
        if model in configs.keys():
            for key in configs[model].keys():
                full_configs[model][key] = configs[model][key]
    return full_configs


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    try:
        dgl.seed(seed)
    except Exception as E:
        print(E)
    # torch.backends.cudnn.deterministic = True
    # os.environ["OMP_NUM_THREADS"] = '1'


class IndexDict(dict):
    def __init__(self):
        super(IndexDict, self).__init__()
        self.count = 0

    def __getitem__(self, item):
        if item not in self.keys():
            super().__setitem__(item, self.count)
            self.count += 1
        return super().__getitem__(item)


def add_new_elements(graph, nodes={}, ndata={}, edges={}, edata={}):
    '''
    :param graph: Dgl
    :param nodes: {ntype: []}
    :param ndata: {ntype: {attr: []}
    :param edges: {(src_type, etype, dst_type): [(src, dst)]
    :param ndata: {etype: {attr: []}
    :return:
    '''
    # etypes = graph.canonical_etypes
    num_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
    for ntype in nodes:
        num_nodes_dict[ntype] = nodes[ntype]
    # etypes.extend(list(edges.keys()))

    relations = {}
    for etype in graph.canonical_etypes:
        src, dst = graph.edges(etype=etype[1])
        relations[etype] = (src, dst)
    for etype in edges:
        relations[etype] = edges[etype]

    # print(relations)
    # print(num_nodes_dict)
    new_g = dgl.heterograph(relations, num_nodes_dict=num_nodes_dict)

    for ntype in graph.ntypes:
        for k, v in graph.nodes[ntype].data.items():
            new_g.nodes[ntype].data[k] = v.detach().clone()
    for etype in graph.etypes:
        for k, v in graph.edges[etype].data.items():
            new_g.edges[etype].data[k] = v.detach().clone()

    for ntype in ndata:
        for attr in ndata[ntype]:
            new_g.nodes[ntype].data[attr] = ndata[ntype][attr]

    for etype in edata:
        for attr in edata[etype]:
            # try:
            new_g.edges[etype].data[attr] = edata[etype][attr]
            # except Exception as e:
            #     print(e)
            #     print(etype, attr)
            #     print(new_g.edges(etype=etype))
            #     print(edata[etype][attr])
            #     raise Exception
    # print(new_g)
    return new_g


def get_norm_adj(sparse_adj):
    I = torch.diag(torch.ones(sparse_adj.shape[0])).to_sparse()
    sparse_adj = sparse_adj + I
    D = torch.diag(torch.sparse.sum(sparse_adj, dim=-1).pow(-1).to_dense())
    # print(I)
    # print(sparse_adj.to_dense())
    return torch.matmul(D, sparse_adj.to_dense()).to_sparse()


def custom_node_unbatch(g):
    # just unbatch the node attrs for cl, not including etypes for efficiency
    node_split = {ntype: g.batch_num_nodes(ntype) for ntype in g.ntypes}
    node_split = {k: dgl.backend.asnumpy(split).tolist() for k, split in node_split.items()}
    print(node_split)
    node_attr_dict = {ntype: {} for ntype in g.ntypes}
    for ntype in g.ntypes:
        for key, feat in g.nodes[ntype].data.items():
            subfeats = dgl.backend.split(feat, node_split[ntype], 0)
            # print(subfeats)
            node_attr_dict[ntype][key] = subfeats
    return node_attr_dict


def get_tsne(embs, file_name, indexes=None):
    perplexity_list = [5, 10, 20, 30, 50]
    embs = embs.astype(np.float32)
    embs = torch.from_numpy(embs)
    embs = F.normalize(embs).numpy()
    lr = max(embs.shape[0] / 12 / 4, 50)
    # lr = 1000
    print('add norm!')
    # print(lr)
    # tsne = TSNE(n_components=2, learning_rate=lr, init='random', perplexity=30)
    for perplexity in perplexity_list:
        tsne = TSNE(n_components=2, learning_rate=50, perplexity=perplexity, init='pca', verbose=1)
        result = tsne.fit_transform(embs)
        print(result.shape)
        # plt.scatter(x=result[:, 0], y=result[:, 1])
        # plt.savefig('./imgs/{}.pdf'.format(model_name), bbox_inches='tight')
        # joblib.dump(result, file_name.format(perplexity))
        torch.save([indexes, result], file_name.format(perplexity))

def get_label(citation):
    if citation < 10:
        return 0
    elif citation >= 100:
        return 2
    else:
        return 1


def get_sampled_index(data):
    print('wip')
    df = pd.DataFrame(data['test'][1])
    df['label'] = df[0].apply(get_label)
    sampled_count = 1000
    print(df[df['label'] == 0])
    cls_0 = df[df['label'] == 0].sample(sampled_count, random_state=123).index
    cls_1 = df[df['label'] == 1].sample(sampled_count, random_state=123).index
    print(cls_0[:5], cls_1[:5])
    cls_2 = df[df['label'] == 2].sample(sampled_count, random_state=123).index
    all_index = list(cls_0) + list(cls_1) + list(cls_2)
    return all_index


def get_selected_data(ids, dataProcessor):
    data = dataProcessor.get_data()
