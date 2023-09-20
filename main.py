import logging

import joblib
import pandas as pd

from data_processor import *
from our_models.FCL import FCLWrapper
from our_models.DCTSGCN import DCTSGCN
from our_models.TSGCN import TSGCN

from utilis.scripts import get_configs, setup_seed, get_sampled_index

LOAD_JOBLIB = True


def get_model(model_name, config, vectors=None, **kwargs):
    model = None
    if model_name == 'TSGCN':
        model = TSGCN(vocab_size=0, embed_dim=config['embed_dim'], num_classes=config['num_classes'], pad_index=0,
                      n_layers=config['n_layers'], time_encoder=config['time_encoder'], etypes=config['etypes'],
                      ntypes=config['ntypes'],
                      time_encoder_layers=config['time_encoder_layers'], time_pooling='sum', hop=config['hop'],
                      time_length=config['time_length'], hidden_dim=config['hidden_dim'], out_dim=config['out_dim'],
                      word2vec=vectors, dropout=config['dropout'], graph_type=config['graph_type'],
                      cut_threshold=config['cut_threshold'], aux_weight=config['aux_weight'],
                      data_source=config['data_source'], **kwargs)
    elif model_name == 'DCTSGCN':
        # print(config['etypes'], **kwargs)
        model = DCTSGCN(vocab_size=0, embed_dim=config['embed_dim'], num_classes=config['num_classes'], pad_index=0,
                        n_layers=config['n_layers'], ntypes=config['ntypes'], etypes=config['etypes'],
                        pred_type=config['pred_type'], time_pooling='sum', hop=config['hop'],
                        encoder_type=config['encoder_type'], gcn_out=config['gcn_out'],
                        edge_sequences=config['edge_sequences'], linear_before_gcn=config['linear_before_gcn'],
                        time_length=config['time_length'], hidden_dim=config['hidden_dim'], out_dim=config['out_dim'],
                        word2vec=vectors, dropout=config['dropout'], graph_type=config['graph_type'],
                        cut_threshold=config['cut_threshold'], aux_weight=config['aux_weight'],
                        data_source=config['data_source'], **kwargs)
    # elif model_name == 'DCTSGCNFCL':
    elif model_name == 'H2CGL':
        # print(config['etypes'], **kwargs)
        encoder = DCTSGCN(vocab_size=0, embed_dim=config['embed_dim'], num_classes=config['num_classes'], pad_index=0,
                          n_layers=config['n_layers'], ntypes=config['ntypes'], etypes=config['etypes'],
                          pred_type=config['pred_type'], time_pooling='sum', hop=config['hop'],
                          encoder_type=config['encoder_type'], gcn_out=config['gcn_out'],
                          edge_sequences=config['edge_sequences'], linear_before_gcn=config['linear_before_gcn'],
                          time_length=config['time_length'], hidden_dim=config['hidden_dim'], out_dim=config['out_dim'],
                          word2vec=vectors, dropout=config['dropout'], graph_type=config['graph_type'],
                          cut_threshold=config['cut_threshold'], aux_weight=config['aux_weight'], hn=config['hn'],
                          hn_method=config['hn_method'],
                          data_source=config['data_source'], **kwargs)
        model = FCLWrapper(encoder=encoder, cl_type=config['cl_type'], aug_type=config['aug_type'], tau=config['tau'],
                           cl_weight=config['cl_weight'], aug_rate=config['aug_rate'], aux_weight=config['aux_weight'],
                           data_source=config['data_source'])
    model.to(model.device)
    return model


def load_data_and_model(data_source, model_name, config, seed=123, norm=False, log=False, log_tag=None, phases=None,
                        **kwargs):
    record_path = './results/{}/'.format(data_source)
    save_path = './checkpoints/{}/'.format(data_source) if config['save_model'] else None
    if config['type'] == 'normal':
        dataProcessor = DataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm, time=config['time'],
                                      model_config=config, log=log)
    elif config['type'] == 'cascade':
        dataProcessor = CascadeDataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm,
                                             max_time=config['time_length'],
                                             time=config['time'], structure=config['structure'],
                                             model_config=config, log=log)
    print(config['tokenizer_type'])
    config['model_name'] = model_name
    dataProcessor.get_tokenizer(config['tokenizer_type'], config['tokenizer_path'])

    config['vocab_size'] = len(dataProcessor.tokenizer.vocab)
    config['pad_idx'] = dataProcessor.tokenizer.vocab[PAD]
    model = get_model(model_name, config, dataProcessor.tokenizer.vectors, **kwargs)
    model.get_criterion(config['criterion'])
    if log_tag:
        file_name = record_path + '{}_{}.log'.format(model.model_name, log_tag)
    else:
        file_name = record_path + '{}.log'.format(model.model_name)
    logging.basicConfig(level=logging.INFO,
                        filename=file_name,
                        filemode='w+',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        force=True)

    if config['use_graph']:
        if ('TSGCN' in model_name) | ('H2' in model_name):
            graph_dict = dataProcessor.load_graphs(config['graph_name'], time_length=config['time_length'],
                                                   phases=phases)
        else:
            train_time, val_time, test_time = dataProcessor.time
            graph_dict = {
                'data': {phase: None for phase in phases},
                'node_trans': json.load(open(dataProcessor.data_cat_path + 'sample_node_trans.json', 'r')),
                'time': {'train': train_time, 'val': val_time, 'test': test_time},
                'all_graphs': None,
                'time_length': config['time_length']
            }
    else:
        graph_dict = {
            'data': {phase: None for phase in phases}}

    dealt_graph_dict = get_graphs(model_name, config, model, dataProcessor, graph_dict)

    emb_name = config.get('emb_name', None)
    if emb_name and dealt_graph_dict:
        dealt_graph_dict['emb_path'] = dataProcessor.data_cat_path + emb_name + '_embs'

    dealt_graph_dict = model.load_graphs(dealt_graph_dict)
    # batch_graphs = dealt_graph_dict if config['batch_graph'] else None
    batch_graphs = dealt_graph_dict if dealt_graph_dict else graph_dict
    dataProcessor.get_dataloader(config['batch_size'], num_workers=config['num_workers'], graph_dict=batch_graphs)

    return dataProcessor, dealt_graph_dict, model, config, save_path, record_path


def train_single(data_source, model_name, config, seed=123, norm=False, log=False, aux=False, **kwargs):
    phases = ['train', 'val', 'test']
    if config['test_interval'] == 0:
        phases.remove('test')
    dataProcessor, dealt_graph_dict, model, config, save_path, record_path = load_data_and_model(
        data_source, model_name, config, seed=seed, norm=norm, log=log, aux=aux, phases=phases, **kwargs)

    model.train_batch(dataloaders=dataProcessor.dataloaders, epochs=config['epochs'],
                      lr=config['lr'], weight_decay=config['weight_decay'], criterion=config['criterion'],
                      optimizer=config['optimizer'],
                      record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                      test_interval=config['test_interval'], best_metric=config['best_metric'], aux=aux)
    if save_path:
        if config['test_interval'] > 0:
            model.get_test_results(dataloader=dataProcessor.dataloaders[-1],
                                   record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                                   best_metric=config['best_metric'], aux=aux)
        else:
            del dataProcessor, dealt_graph_dict
            test_results(data_source, model_name, config, seed=seed, norm=norm, log=log, aux=aux, **kwargs)


def test_results(data_source, model_name, config, seed=123, norm=False, log=False, aux=False, **kwargs):
    phases = ['test']
    dataProcessor, dealt_graph_dict, model, config, save_path, record_path = load_data_and_model(
        data_source, model_name, config, seed=seed, norm=norm, log=log, log_tag='test', phases=phases, **kwargs)

    model.get_test_results(dataloader=dataProcessor.dataloaders[0],
                           record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                           best_metric=config['best_metric'], aux=aux)


def show_results(data_source, model_name, config, seed=123, norm=False, log=False, aux=False, show_phase='test',
                 show='weight', **kwargs):
    phases = [show_phase]
    dataProcessor, dealt_graph_dict, model, config, save_path, record_path = load_data_and_model(
        data_source, model_name, config, seed=seed, norm=norm, log=log, log_tag='show', phases=phases, **kwargs)
    model.get_show_results(dataloader=dataProcessor.dataloaders[0],
                           record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                           best_metric=config['best_metric'], aux=aux, phase=show_phase, show=show)


def selected_results(data_source, model_name, config, seed=123, norm=False, log=False, aux=False, show_phase='test',
                     **kwargs):
    phases = [show_phase]
    dataProcessor, dealt_graph_dict, model, config, save_path, record_path = load_data_and_model(
        data_source, model_name, config, seed=seed, norm=norm, log=log, log_tag='show', phases=phases, **kwargs)
    model.get_show_results(dataloader=dataProcessor.dataloaders[0],
                           record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                           best_metric=config['best_metric'], aux=aux, phase=show_phase, show='graph')


def get_graphs(model_name, config, model, dataProcessor, graph_dict, force=False, selected=False):
    if model_name == 'H2CGL':
        model_name = 'DCTSGCNFCL'
    if config['use_graph'] == 'one-time':
        # graph_dict = dataProcessor.load_graphs(config['graph_name'])
        dealt_graph_dict = model.deal_graphs(graph_dict, data_path=dataProcessor.data_cat_path + 'split_data',
                                             log_path='./results/{}/{}.log'.format(dataProcessor.data_source,
                                                                                   model_name))
    elif config['use_graph'] == 'repeatedly':
        # graphs_path = './checkpoints/{}/{}_graphs'.format(data_source, model.model_name)
        graph_type = config.get('graph_type', config['graph_name'].split('_')[-1])
        # if graph_type != 'vector':
        #     graphs_path = './checkpoints/{}/{}_graphs_{}'.format(dataProcessor.data_source, model_name, graph_type)
        # else:
        #     graphs_path = './checkpoints/{}/{}_graphs'.format(dataProcessor.data_source, model_name)
        # graphs_path = './checkpoints/{}/{}_graphs_{}'.format(dataProcessor.data_source, model_name, graph_type)
        graphs_path = './checkpoints/{}/{}_graphs'.format(dataProcessor.data_source, model_name)
        # if model_name.endswith('TSGCN'):
        #     graphs_path = './checkpoints/{}/{}_graphs'.format(data_source, 'TSGCN')
        print(model_name)

        if 'HTSGCN' in model_name:
            graphs_path = graphs_path.replace(model_name, 'HTSGCN')
        if 'CTSGCN' in model_name:
            graphs_path = graphs_path.replace(model_name, 'CTSGCN')

        if (os.path.exists(graphs_path)) & (not force):
            try:
                # if model_name.endswith('TSGCN'):
                #     dealt_graph_dict = joblib.load(graphs_path)
                if model_name == 'TSGCN':
                    # graphs_path = './checkpoints/{}/{}_graphs'.format(data_source, 'TSGCN')
                    # dealt_graph_dict = dataProcessor.load_graphs(config['graph_name'])
                    dealt_graph_dict = graph_dict
                    for phase in dealt_graph_dict['data']:
                        dealt_graph_dict['data'][phase] = None
                    split_data = torch.load(dataProcessor.data_cat_path + 'split_data')
                    # ids
                    dealt_graph_dict['selected_papers'] = {}
                    # print(dealt_graph_dict['data'])
                    for phase in dealt_graph_dict['data']:
                        dealt_graph_dict['selected_papers'][phase] = split_data[phase][0]
                    del split_data
                    dealt_graph_dict['graphs_path'] = graphs_path
                    # dealt_graph_dict['all_trans_index'] = json.load(open(graphs_path + '_trans.json', 'r'))
                    # # ego_graphs
                    # dealt_graph_dict['all_ego_graphs'] = {}
                    # for pub_time in tqdm(dealt_graph_dict['all_graphs']):
                    #     if LOAD_JOBLIB:
                    #         dealt_graph_dict['all_ego_graphs'][pub_time] = joblib.load(
                    #             graphs_path + '_{}.job'.format(pub_time))
                    #     else:
                    #         dealt_graph_dict['all_ego_graphs'][pub_time] = torch.load(graphs_path + '_' + str(pub_time))
                elif ('CTSGCN' in model_name) | ('HTSGCN' in model_name):
                    # dealt_graph_dict = dataProcessor.load_graphs(config['graph_name'])
                    print(graphs_path)
                    dealt_graph_dict = graph_dict
                    phases = list(dealt_graph_dict['data'].keys())
                    dealt_graph_dict['data'] = {}
                    for phase in phases:
                        if ('CL' in model_name) and (phase == 'train') and (model.aug_type == 'cg'):
                            print('load fixed CL data!!!')
                            dealt_graph_dict['data'][phase] = joblib.load(graphs_path + '_cl_{}_{}.job'.
                                                                          format(model.aug_type, model.aug_rate))
                        elif selected:
                            dealt_graph_dict['data'][phase] = joblib.load(graphs_path + '_selected.job')
                        else:
                            dealt_graph_dict['data'][phase] = joblib.load(graphs_path + '_{}.job'.format(phase))
                else:
                    # del graph_dict
                    # dealt_graph_dict = torch.load(graphs_path)
                    # dealt_graph_dict = joblib.load(graphs_path)
                    dealt_graph_dict = graph_dict
                    phases = list(dealt_graph_dict['data'].keys())
                    for phase in phases:
                        dealt_graph_dict['data'][phase] = joblib.load(graphs_path + '_{}.job'.format(phase))
            except Exception as e:
                print(e)
                dealt_graph_dict = joblib.load(graphs_path)
            print('load graphs successfully!')
            if 'time' not in dealt_graph_dict:
                dealt_graph_dict['time'] = dataProcessor.get_time_dict()
            if 'all_graphs' in dealt_graph_dict:
                if dealt_graph_dict['all_graphs'] is None:
                    print('adding all graphs')
                    # graph_dict = dataProcessor.load_graphs(config['graph_name'])
                    dealt_graph_dict['all_graphs'] = graph_dict['all_graphs']
                    del graph_dict
        else:
            # graph_dict = dataProcessor.load_graphs(config['graph_name'])
            # if model_name.endswith('TSGCN'):
            #     graphs_path = './checkpoints/{}/{}_graphs'.format(dataProcessor.data_source, 'TSGCN')
            # if ('CTSGCN' in model_name) | ('HTSGCN' in model_name) | ('CSTSGCN' in model_name):
            print(model_name, graphs_path)
            if model_name.endswith('TSGCN'):
                graphs_path = graphs_path.replace(model_name, 'TSGCN')
            dealt_graph_dict = model.deal_graphs(graph_dict, data_path=dataProcessor.data_cat_path + 'split_data',
                                                 path=graphs_path,
                                                 log_path='./results/{}/{}_dealing.log'.format(
                                                     dataProcessor.data_source, model_name))
        dealt_graph_dict['data_source'] = dataProcessor.data_source
    else:
        dealt_graph_dict = None
    # if dataProcessor.embs:
    #     dealt_graph_dict['embs'] = dataProcessor.embs
    # print(dealt_graph_dict['all_graphs'])

    return dealt_graph_dict


def train_cl(data_source, model_name, config, seed=123, norm=False):
    if config['type'] == 'normal':
        dataProcessor = DataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm, time=config['time'],
                                      model_config=config)
    elif config['type'] == 'cascade':
        dataProcessor = CascadeDataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm,
                                             time=config['time'], structure=config['structure'], model_config=config)
    print(config['tokenizer_type'])
    # dataProcessor.get_tokenizer(config['tokenizer_type'], config['tokenizer_path'])
    dataProcessor.get_cl_dataloader(config['batch_size'])

    record_path = './results/{}/'.format(data_source)
    save_path = './checkpoints/{}/'.format(data_source)

    config['vocab_size'] = 0
    config['pad_idx'] = 0
    model = get_model(model_name, config, None)
    model.train_batch(dataloaders=dataProcessor.dataloaders, epochs=config['epochs'],
                      lr=config['lr'], weight_decay=config['weight_decay'], criterion=config['criterion'],
                      optimizer=config['optimizer'],
                      record_path=record_path, save_path=save_path)


def get_model_graph_data(data_source, model_name, config, seed=123, norm=False):
    if config['type'] == 'normal':
        dataProcessor = DataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm, time=config['time'],
                                      model_config=config)
    elif config['type'] == 'cascade':
        dataProcessor = CascadeDataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm,
                                             time=config['time'], structure=config['structure'], model_config=config)
    print(config['tokenizer_type'])
    dataProcessor.get_tokenizer(config['tokenizer_type'], config['tokenizer_path'])
    # dataProcessor.get_dataloader(config['batch_size'])

    # record_path = './results/{}/'.format(data_source)
    # save_path = './checkpoints/{}/'.format(data_source)

    config['vocab_size'] = len(dataProcessor.tokenizer.vocab)
    config['pad_idx'] = dataProcessor.tokenizer.vocab[PAD]
    model = get_model(model_name, config, dataProcessor.tokenizer.vectors)
    print('time_length:', config['time_length'])
    graph_dict = dataProcessor.load_graphs(config['graph_name'], time_length=config['time_length'])
    get_graphs(model_name, config, model, dataProcessor, graph_dict, force=True)
    # if config['use_graph'] == 'one-time':
    #     graph_dict = dataProcessor.load_graphs(config['graph_name'])
    #     dealt_graph_dict = model.deal_graphs(graph_dict, data_path=dataProcessor.data_cat_path + 'split_data',
    #                                          log_path='./results/{}/{}_dealing.log'.format(dataProcessor.data_source,
    #                                                                                        model_name))
    # elif config['use_graph'] == 'repeatedly':
    #     graphs_path = './checkpoints/{}/{}_graphs'.format(data_source, model.model_name)
    #     graph_dict = dataProcessor.load_graphs(config['graph_name'])
    #     dealt_graph_dict = model.deal_graphs(graph_dict, data_path=dataProcessor.data_cat_path + 'split_data',
    #                                          path=graphs_path,
    #                                          log_path='./results/{}/{}_dealing.log'.format(dataProcessor.data_source,
    #                                                                                        model_name))


def get_best_values(data_source, metric, ascending=True):
    data_path = './results/{}/'.format(data_source)
    records = [file for file in os.listdir(data_path) if file.endswith('.csv')]
    temp_list = []
    for record in records:
        print(record)
        df = pd.read_csv((data_path + record))
        # print(df.columns)
        try:
            df = df.sort_values(by=metric, ascending=ascending).head(1)
            df['model'] = [record.split('_records')[0]]
            df = df[['model'] + list(df.columns[:-1])]
            # df.columns = ['model'] + list(df.columns[:-1])
            temp_list.append(df)
            print(df)
        except Exception as e:
            continue
    result_df = pd.DataFrame(columns=temp_list[0].columns)
    for record in temp_list:
        result_df = result_df.append(record)
    result_df.to_excel(data_path + 'all_{}_best.xlsx'.format(metric))


def get_fixed_cl_data(data_source, model_name, config, seed=123, norm=False):
    if config['type'] == 'normal':
        dataProcessor = DataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm, time=config['time'],
                                      model_config=config)
    elif config['type'] == 'cascade':
        dataProcessor = CascadeDataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm,
                                             time=config['time'], structure=config['structure'], model_config=config)
    print(config['tokenizer_type'])
    dataProcessor.get_tokenizer(config['tokenizer_type'], config['tokenizer_path'])
    # dataProcessor.get_dataloader(config['batch_size'])

    record_path = './results/{}/'.format(data_source)
    # save_path = './checkpoints/{}/'.format(data_source)

    config['vocab_size'] = len(dataProcessor.tokenizer.vocab)
    config['pad_idx'] = dataProcessor.tokenizer.vocab[PAD]
    model = get_model(model_name, config, dataProcessor.tokenizer.vectors)

    logging.basicConfig(level=logging.INFO,
                        filename=record_path + '{}_cl_data_{}.log'.format(model_name, model.aug_type),
                        filemode='w+',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        force=True)

    graphs_path = './checkpoints/{}/{}_graphs'.format(dataProcessor.data_source, model_name)
    # if model_name.endswith('TSGCN'):
    #     graphs_path = './checkpoints/{}/{}_graphs'.format(data_source, 'TSGCN')
    print(model_name)

    if 'HTSGCN' in model_name:
        graphs_path = graphs_path.replace(model_name, 'HTSGCN')
    if 'CTSGCN' in model_name:
        graphs_path = graphs_path.replace(model_name, 'CTSGCN')

    train_graphs = joblib.load(graphs_path + '_train.job')
    dealt_graphs = {'data': {
        'train': train_graphs
    }}
    dealt_graphs = model.encoder.aug_graphs_plus(dealt_graphs, model.aug_configs, model.aug_methods)
    joblib.dump(dealt_graphs['data']['train'], graphs_path + '_cl_{}_{}.job'.format(model.aug_type, model.aug_rate))


def get_selected_data(data_source, model_name, config, seed=123, norm=False):
    if config['type'] == 'normal':
        dataProcessor = DataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm, time=config['time'],
                                      model_config=config)
    elif config['type'] == 'cascade':
        dataProcessor = CascadeDataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm,
                                             time=config['time'], structure=config['structure'], model_config=config)
    print(config['tokenizer_type'])
    dataProcessor.get_tokenizer(config['tokenizer_type'], config['tokenizer_path'])
    # dataProcessor.get_dataloader(config['batch_size'])

    record_path = './results/{}/'.format(data_source)
    # save_path = './checkpoints/{}/'.format(data_source)

    config['vocab_size'] = len(dataProcessor.tokenizer.vocab)
    config['pad_idx'] = dataProcessor.tokenizer.vocab[PAD]
    model = get_model(model_name, config, dataProcessor.tokenizer.vectors)

    logging.basicConfig(level=logging.INFO,
                        filename=record_path + '{}_selected_data.log'.format(model_name),
                        filemode='w+',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        force=True)

    graphs_path = './checkpoints/{}/{}_graphs'.format(dataProcessor.data_source, model_name)
    # if model_name.endswith('TSGCN'):
    #     graphs_path = './checkpoints/{}/{}_graphs'.format(data_source, 'TSGCN')
    print(model_name)

    if 'CTSGCN' in model_name:
        graphs_path = graphs_path.replace(model_name, 'CTSGCN')

    test_graphs = joblib.load(graphs_path + '_test.job')
    selected_ids = list(dataProcessor.get_selected_ids())
    paper_trans = json.load(open(dataProcessor.data_cat_path + 'sample_node_trans.json', 'r'))['paper']
    selected_ids = [paper_id for paper_id in selected_ids if paper_id in paper_trans]
    print(len(selected_ids))
    selected_ids = [paper_trans[paper_id] for paper_id in selected_ids]
    combined_graphs_list, trans_index = test_graphs
    selected_graphs_list = []
    true_ids = []
    for paper_id in selected_ids:
        if paper_id in trans_index:
            true_ids.append(paper_id)
            selected_graphs_list.append(combined_graphs_list[trans_index[paper_id]])
    print(len(selected_graphs_list))
    # selected_trans_index = dict(zip(selected_ids, range(len(selected_ids))))
    selected_trans_index = dict(zip(true_ids, range(len(true_ids))))
    joblib.dump([selected_graphs_list, selected_trans_index], graphs_path + '_selected.job')


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')

    parser.add_argument('--phase', default='test_results', help='the function name.')
    parser.add_argument('--ablation', default=None, help='the ablation modules.')
    parser.add_argument('--data_source', default='h_pubmed', help='the data source.')
    parser.add_argument('--norm', default=False, help='the data norm.')
    parser.add_argument('--log', default=True, help='apply log on the objective.')
    parser.add_argument('--mode', default=None, help='the model mode.')
    parser.add_argument('--type', default='o', help='the model type.')
    parser.add_argument('--seed', default=123, help='the data seed.')
    parser.add_argument('--model_seed', default=123, help='the model seed.')
    parser.add_argument('--model', default='H2CGL', help='the selected model for other methods.')
    parser.add_argument('--model_path', default=None, help='the selected model for deep analysis.')
    parser.add_argument('--pred_type', default='snapshot', help='how to get the final emb.')
    parser.add_argument('--graph_type', default='vector', help='the graph type to learn.')
    parser.add_argument('--emb_type', default='bw', help='the graph type to learn.')
    parser.add_argument('--cl_type', default='label_aug_hard_negative', help='the cl type to learn.')
    parser.add_argument('--aug_type', default='cg', help='the aug type to learn.')
    parser.add_argument('--aug_rate', default=0.1, help='the aug type to learn.')
    parser.add_argument('--encoder_type', default='CGIN+RGAT', help='the gcn encoder type to learn.')
    parser.add_argument('--gcn_out', default='mean', help='how to get the node emb.')
    parser.add_argument('--n_layers', default=None, help='the layers of gcn.')
    parser.add_argument('--edge_seq', default=None, help='the edge sequences.')
    parser.add_argument('--inter_mode', default='attn', help='the edge sequences.')
    parser.add_argument('--time_length', default=None, help='the edge sequences.')
    parser.add_argument('--lr', default=None, help='the layers of gcn.')
    parser.add_argument('--aux', default=True, help='add the aux classification task.')
    parser.add_argument('--cl_w', default=None, help='cl weight.')
    parser.add_argument('--aux_w', default=None, help='aux weight.')
    parser.add_argument('--hn', default=2, help='hard negative samples.')
    parser.add_argument('--hn_method', default='co_cite', help='hard negative sampling method.')
    parser.add_argument('--tau', default=None, help='hard negative sampling method.')
    parser.add_argument('--oargs', default=None, help='other args to the model.')
    parser.add_argument('--show', default='weight', help='other args to the model.')

    args = parser.parse_args()
    print('args', args)
    print('data_seed', args.seed)
    # setup_seed(int(args.seed))
    MODEL_SEED = int(args.model_seed)
    setup_seed(MODEL_SEED)
    print('model_seed', MODEL_SEED)

    # model_list = ['DCTSGCN', 'DCTSGCNFCL']
    model_list = ['H2GCN', 'H2GIN', 'H2CGL']
    # model_list = ['TSGCN', 'DCTSGCN', 'H2CGL']
    cl_model_list = []
    data_source = args.data_source
    if args.data_source.startswith('h_'):
        data_source = args.data_source.replace('h_', '')
        default_config = get_configs(data_source, [])['default']
        train_time = default_config['time'][0]
        replace_dict = {'time': [train_time, train_time + 2, train_time + 4]}
    else:
        replace_dict = None
    configs = get_configs(data_source, model_list + cl_model_list, replace_dict)
    # configs['CSTSGCN'] = configs['CTSGCN']

    if args.phase in model_list + cl_model_list:
        cur_model = args.phase
    elif args.model in model_list + cl_model_list:
        cur_model = args.model
    else:
        print('what are you doing')
        raise Exception
    print(cur_model)
    model_config = configs[cur_model]
    model_config['data_source'] = args.data_source

    # =============================================== args ===========================================
    model_config['graph_type'] = None
    model_config['aug_type'] = None
    if args.graph_type:
        model_config['graph_name'] = '_'.join(model_config['graph_name'].split('_')[:-1]
                                              + [args.graph_type.split('+')[0]])
        model_config['graph_type'] = args.graph_type
    if args.emb_type:
        model_config['emb_name'] = '_'.join(model_config['graph_name'].split('_')[:-1]
                                            + [args.emb_type.split('+')[0]])
        model_config['emb_type'] = args.emb_type
    if args.pred_type:
        model_config['pred_type'] = args.pred_type
    if args.cl_type:
        model_config['cl_type'] = args.cl_type
    if args.encoder_type:
        model_config['encoder_type'] = args.encoder_type
        # if args.encoder_type == 'CGIN':
        #     model_config['lr'] = 1e-5
    if args.aug_type:
        model_config['aug_type'] = args.aug_type
    if args.aug_rate:
        model_config['aug_rate'] = float(args.aug_rate)
    if args.gcn_out:
        model_config['gcn_out'] = args.gcn_out
    if args.n_layers:
        model_config['n_layers'] = int(args.n_layers)
    if args.lr:
        model_config['lr'] = float(args.lr)
    if args.edge_seq:
        model_config['edge_sequences'] = args.edge_seq
    if args.inter_mode:
        model_config['inter_mode'] = args.inter_mode
    if args.time_length:
        model_config['time_length'] = int(args.time_length)
    if args.hn:
        model_config['hn'] = int(args.hn)
    else:
        model_config['hn'] = args.hn
    model_config['hn_method'] = args.hn_method
    if args.tau:
        model_config['tau'] = float(args.tau)
    if args.cl_w:
        model_config['cl_weight'] = float(args.cl_w)
    if args.aux_w:
        model_config['aux_weight'] = float(args.aux_w)
    if args.type:
        model_config['model_type'] = args.type

    if args.graph_type == 'sbert':
        model_config['embed_dim'] = 384
    if args.emb_type == 'sbert':
        model_config['bert_dim'] = 384
    else:
        model_config['bert_dim'] = 768

    # special args
    print(args.oargs)
    if args.oargs:
        if 'lbg' in args.oargs:
            model_config['linear_before_gcn'] = True
        int_list = [arg.split('=') for arg in args.oargs.split(',') if '=' in arg]
        for key, value in int_list:
            model_config[key] = int(value)
        str_list = [arg.split(':') for arg in args.oargs.split(',') if ':' in arg]
        for key, value in str_list:
            model_config[key] = value

    # =============================================== args ===========================================

    if args.phase == 'test':
        # get_best_values('dblp', 'val_mae')
        # test(args.data_source, args.model, configs[args.model], args.seed, args.norm)
        get_model('CSTSGCN', configs['CSTSGCN'])
    elif args.phase in model_list:
        train_single(args.data_source, args.phase, model_config, args.seed, args.norm, args.log, args.aux,
                     oargs=args.oargs)
    elif args.phase in cl_model_list:
        print(model_config)
        # print(model_config['structure'])
        train_cl(args.data_source, args.phase, model_config, args.seed, args.norm)
    elif args.phase == 'get_model_graph_data':
        get_model_graph_data(args.data_source, args.model, model_config, args.seed, args.norm)
    elif args.phase == 'test_results':
        test_results(args.data_source, args.model, model_config, args.seed, args.norm, args.log, args.aux,
                     oargs=args.oargs)
    elif args.phase == 'show_results':
        show_results(args.data_source, args.model, model_config, args.seed, args.norm, args.log, args.aux,
                     oargs=args.oargs, result=args.show)
    elif args.phase == 'selected_results':
        selected_results(args.data_source, args.model, model_config, args.seed, args.norm, args.log, args.aux,
                         oargs=args.oargs)
    elif args.phase == 'get_cl_data':
        get_fixed_cl_data(args.data_source, args.model, model_config, args.seed, args.norm)
    elif args.phase == 'get_selected_data':
        get_selected_data(args.data_source, args.model, model_config, args.seed, args.norm)

    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done main!')
