import argparse
import datetime
import json
import math
import os
import random
import logging

import joblib
import numpy as np
import pandas as pd

from torch import nn
import time
import torch
import torch.nn.functional as F
import re

from utilis.scripts import eval_result, result_format, eval_aux_results, aux_result_format


class BaseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_index, word2vec=None, dropout=0.5,
                 model_path=None, **kwargs):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not model_path:
            if word2vec is not None:
                # print(word2vec)
                self.embedding = nn.Embedding.from_pretrained(word2vec, freeze=False)
            elif vocab_size > 0:
                self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False, padding_idx=pad_index)
            else:
                self.embedding = None
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.model_name = 'base_model'
        self.adaptive_lr = False
        self.warmup = False
        self.T = kwargs['T'] if 'T' in kwargs.keys() else 400
        self.seed = None
        self.graphs_data = None
        self.node_trans = None
        self.other_loss = 0
        self.epoch = 0
        self.relu_out = None
        self.test_interval = None
        if 'seed' in kwargs.keys():
            self.seed = kwargs['seed']
        if self.warmup:
            print('warm up T', self.T)
        self.aux_weight = kwargs.get('aux_weight', 0.5)
        self.aux_criterion = nn.CrossEntropyLoss()
        self.cut_threshold = kwargs.get('cut_threshold', [0, 10])
        print(self.cut_threshold)
        self.eval_interval = kwargs.get('eval_interval', 10)

    def init_weights(self):
        print('init')

    def forward(self, content, lengths, masks, ids, graph, times, **kwargs):
        # return predicted values and classes
        return 'forward'

    def predict(self, content, lengths, masks, ids, graph, times, **kwargs):
        return self(content, lengths, masks, ids, graph, times, **kwargs)

    def get_group_parameters(self, lr=0):
        return self.parameters()

    def train_model(self, dataloader, epoch, criterion, optimizer, graph=None, aux=False):
        self.train()
        total_acc, total_count = 0, 0
        log_interval = 100
        start_time = time.time()
        loss_list = []
        all_predicted_values = []
        all_true_values = []
        teacher_forcing_ratio = 0.5
        all_loss = 0
        all_aux_loss = 0
        all_predicted_aux_values = []
        all_true_aux_values = []

        cur_graphs = None
        if graph:
            # load graphs_data
            self.node_trans = graph['node_trans']
            # self.graph = graph['data'].to(self.device)
            # self.graphs_data = graph['data']
            # cur_graphs = self.graphs_data['train']
            cur_graphs = graph['data']['train']
            cur_graphs = self.graph_to_device(cur_graphs)

        for idx, (x, values, lengths, masks, ids, times, blocks) in enumerate(dataloader):
            # loss = 0
            self.other_loss = 0  # refresh other loss for each epoch
            if type(x) == torch.Tensor:
                x = x.to(self.device)
            else:
                x = [content.to(self.device) for content in x]
            # print('io-time:', time.time()-start_time)
            values = values.to(self.device)
            lengths = lengths.to(self.device)
            masks = masks.to(self.device)
            if blocks:
                cur_graphs = blocks
            # print('io-time:', time.time() - start_time)

            # encoder_optimizer.zero_grad()
            # decoder_optimizer.zero_grad()
            optimizer.zero_grad()

            # if self.graphs_data:
            if graph:
                ids = torch.tensor(list(map(lambda x: self.node_trans['paper'][x], ids)))
                # cur_graphs = self.graphs_data['train']
            predicted_values, aux_output = self(x, lengths, masks, ids, cur_graphs, times)
            if aux:
                new_gt = self.get_new_gt(values)
                # predicted_values, aux_output = predicted_values
                aux_loss = self.get_aux_loss(aux_output, new_gt)
                all_aux_loss = all_aux_loss + aux_loss.item()
                self.other_loss = self.other_loss + aux_loss
                # self.other_loss = aux_loss
                predicted_aux_values = self.get_aux_pred(aux_output)

                all_predicted_aux_values.append(predicted_aux_values.detach().cpu())
                all_true_aux_values.append(new_gt.detach().cpu())
            # predicted_values = self(x, lengths, masks, ids, graph)
            true_values = values.unsqueeze(dim=-1)
            if self.relu_out:
                # print(predicted_values)
                # print((true_values + 1).log())
                predicted_values = self.relu_out(predicted_values)
            if dataloader.log:
                # print((true_values + 1).log())
                loss = criterion(predicted_values, (true_values + 1).log())
            else:
                loss = criterion(predicted_values, (true_values - dataloader.mean) / dataloader.std)
            if self.other_loss:
                # print('adding other loss')
                loss = loss + self.other_loss
                # loss = self.other_loss + loss

            loss.backward()
            # if self.aux_criterion:
            #     loss.backward(retain_graph=True)
            #     aux_loss.backward()
            # else:
            #     loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            # if self.warmup:
            #     self.scheduler.step()
            # encoder_optimizer.step()
            # decoder_optimizer.step()
            optimizer.step()

            if dataloader.log:
                # print(predicted_values.detach().cpu().exp())
                all_predicted_values.append(predicted_values.detach().cpu().exp() - 1)
            else:
                all_predicted_values.append(predicted_values.detach().cpu() * dataloader.std + dataloader.mean)
            all_true_values.append(true_values.detach().cpu())
            # print(predicted_values.detach().cpu() * dataloader.std + dataloader.mean)
            # print(true_values.detach().cpu())

            # print_loss = loss.item()
            print_loss = loss.sum(dim=0).item()
            all_loss += print_loss
            rmse = math.sqrt(print_loss)

            if idx % log_interval == 0 and idx > 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| RMSE {:8.3f} | loss {:8.3f}'.format(epoch, idx, len(dataloader),
                                                             rmse, print_loss))
                # epoch_log.write('| epoch {:3d} | {:5d}/{:5d} batches '
                #                 '| accuracy {:8.3f} | loss {:8.3f}\n'.format(epoch, idx, len(dataloader),
                #                                                              total_acc / total_count, loss.item()))
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches '
                             '| RMSE {:8.3f} | loss {:8.3f}'.format(epoch, idx, len(dataloader),
                                                                    rmse, print_loss))
                total_acc, total_count = 0, 0
        elapsed = time.time() - start_time
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, elapsed))
        logging.info('-' * 59)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, elapsed))
        all_predicted_values = torch.cat(all_predicted_values, dim=0).numpy()
        all_true_values = torch.cat(all_true_values, dim=0).numpy()
        # print(all_predicted_values.shape)
        avg_loss = all_loss / len(dataloader)
        print(len(dataloader))
        results = [avg_loss] + eval_result(all_true_values, all_predicted_values)
        format_str = result_format(results)

        for line in format_str.split('\n'):
            logging.info(line)

        if aux:
            print('=' * 89)
            logging.info('=' * 89)
            avg_aux_loss = all_aux_loss / len(dataloader)
            all_predicted_aux_values = torch.cat(all_predicted_aux_values, dim=0).numpy()
            all_true_aux_values = torch.cat(all_true_aux_values, dim=0).numpy()
            aux_results = [avg_aux_loss] + eval_aux_results(all_true_aux_values, all_predicted_aux_values)
            aux_format_str = aux_result_format(aux_results, phase='train')

            for line in aux_format_str.split('\n'):
                logging.info(line)

            print('=' * 89)
            logging.info('=' * 89)
            results += aux_results

        return results

    # def get_params(self):
    #     decoder_params_set = list(map(id, self.decoder.parameters()))
    #     decoder_params = self.decoder.parameters()
    #     encoder_params = filter(lambda p: id(p) not in decoder_params_set, self.parameters())
    #     return encoder_params, decoder_params

    def deal_graphs(self, graphs, data_path=None, path=None, log_path=None):
        # for different model, the graphs_data are different
        return graphs

    def load_graphs(self, graphs):
        return graphs

    def graph_to_device(self, graphs):
        return graphs

    def get_aux_loss(self, pred, gt):
        # print(pred.shape)
        # print(gt.shape)
        # print(pred)
        # print(gt)
        aux_loss = self.aux_criterion(pred, gt) * self.aux_weight
        return aux_loss

    def get_aux_pred(self, pred):
        # return pred.argmax(dim=-1)
        aux_pred = torch.softmax(pred, dim=-1)
        return aux_pred

    def get_new_gt(self, gt):
        new_gt = torch.zeros_like(gt, dtype=torch.long)
        mid_range = gt > self.cut_threshold[0]
        last_range = gt > self.cut_threshold[1]
        new_gt[mid_range ^ last_range] = 1
        new_gt[last_range] = 2
        return new_gt

    def get_optimizer(self, lr, optimizer, weight_decay=1e-3):
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.get_group_parameters(lr), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'ADAM':
            optimizer = torch.optim.Adam(self.get_group_parameters(lr), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'ADAMW':
            optimizer = torch.optim.AdamW(self.get_group_parameters(lr), lr=lr)
        else:
            optimizer = torch.optim.SGD(self.get_group_parameters(lr), lr=lr, weight_decay=weight_decay)
        return optimizer

    def get_criterion(self, criterion):
        if criterion == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        elif criterion == 'MSE':
            criterion = nn.MSELoss()
        self.criterion = criterion
        return criterion

    def train_batch(self, dataloaders, epochs, lr=1e-3, weight_decay=1e-4, criterion='MSE', optimizer='ADAM',
                    scheduler=False, record_path=None, save_path=None, graph=None, test_interval=1, best_metric='male',
                    aux=False):
        final_results = []
        train_dataloader, val_dataloader, test_dataloader = dataloaders

        criterion = self.get_criterion(criterion)
        # encoder_optimizer, decoder_optimizer= self.get_optimizer(lr, optimizer)
        optimizer = self.get_optimizer(lr, optimizer, weight_decay=weight_decay)
        if aux:
            self.model_name = self.model_name + '_aux'
        if self.aux_weight != 0.5:
            self.model_name = self.model_name + '_aw{}'.format(self.aux_weight)

        records_logger = logging.getLogger('records')
        # records_logger.setLevel(logging.DEBUG)
        if self.seed:
            print('{}_records_{}.csv'.format(self.model_name, self.seed))
            # fw = open(record_path + '{}_records_{}.csv'.format(self.model_name, self.seed), 'w')
            # fh = logging.FileHandler(record_path + '{}_records_{}.csv'.format(self.model_name, self.seed), mode='w+')
            record_name = '{}_records_{}.csv'.format(self.model_name, self.seed)

        else:
            # fw = open(record_path + '{}_records.csv'.format(self.model_name), 'w')
            # fh = logging.FileHandler(record_path + '{}_records.csv'.format(self.model_name), mode='w+')
            record_name = '{}_records.csv'.format(self.model_name)
        # if aux:
        #     record_name = record_name.replace('.csv', '_aux.csv')

        fh = logging.FileHandler(record_path + record_name, mode='w+')
        records_logger.handlers = []
        records_logger.addHandler(fh)
        metrics = 'loss, acc, mae, r2, mse, rmse, Mrse, mrse, male, log_r2, msle, rmsle, smape, mape'.split(', ')
        if aux:
            metrics += 'aux_loss, acc, roc_auc, log_loss_value, ' \
                       'micro_prec, micro_recall, micro_f1, ' \
                       'macro_prec, macro_recall, macro_f1'.split(', ')
        raw_header = ',' + ','.join(['{}_' + metric for metric in metrics])
        records_logger.warning('epoch' + raw_header.replace('{}', 'train')
                               + raw_header.replace('{}', 'val')
                               + raw_header.replace('{}', 'test'))

        # logging.basicConfig(level=logging.INFO,
        #                     filename=record_path + '{}.log'.format(self.model_name),
        #                     filemode='w+',
        #                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        #                     force=True)
        # graph = self.load_graphs(graph)
        interval_best = None
        all_interval_best = None
        eval_interval = test_interval if test_interval > 0 else self.eval_interval
        self.test_interval = test_interval
        self.record_path = record_path
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            logging.basicConfig(level=logging.INFO,
                                filename=record_path + '{}_epoch_{}.log'.format(self.model_name, epoch),
                                filemode='w+',
                                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                force=True)
            train_results = self.train_model(train_dataloader, epoch, criterion, optimizer, graph, aux)
            val_results = self.evaluate(val_dataloader, graph=graph, aux=aux)
            cur_value = val_results[metrics.index(best_metric)]
            # print(cur_value)

            if interval_best:
                if best_metric in ['acc', 'r2', 'log_r2']:
                    if cur_value >= interval_best:
                        interval_best = cur_value
                        print('new checkpoint! epoch {}!'.format(epoch))
                        if save_path:
                            self.save_model(save_path + '{}_{}.pkl'.format(self.model_name, (epoch - 1) // eval_interval))
                else:
                    if cur_value <= interval_best:
                        interval_best = cur_value
                        print('new checkpoint! epoch {}!'.format(epoch))
                        if save_path:
                            self.save_model(save_path + '{}_{}.pkl'.format(self.model_name, (epoch - 1) // eval_interval))
            else:
                interval_best = cur_value
                print('new checkpoint! epoch {}!'.format(epoch))
                if save_path:
                    self.save_model(save_path + '{}_{}.pkl'.format(self.model_name, (epoch - 1) // eval_interval))

            if all_interval_best is None:
                all_interval_best = cur_value
                if save_path:
                    self.save_model(save_path + '{}.pkl'.format(self.model_name))
            else:
                if best_metric in ['acc', 'r2', 'log_r2']:
                    if cur_value >= all_interval_best:
                        all_interval_best = cur_value
                        print('global new checkpoint! epoch {}!'.format(epoch))
                        if save_path:
                            self.save_model(save_path + '{}.pkl'.format(self.model_name))
                else:
                    if cur_value <= all_interval_best:
                        all_interval_best = cur_value
                        print('global new checkpoint! epoch {}!'.format(epoch))
                        if save_path:
                            self.save_model(save_path + '{}.pkl'.format(self.model_name))

            # 固定频率测试减少训练时间
            if epoch % eval_interval == 0:
                if test_interval > 0:
                    test_results = self.test(test_dataloader, graph=graph, aux=aux)
                else:
                    test_results = [0] * len(val_results)
                interval_best = None
            else:
                test_results = [0] * len(val_results)
            all_results = train_results + val_results + test_results
            # print(all_results)
            # fw.write(','.join([str(epoch)] + [str(round(x, 6)) for x in all_results]) + '\n')
            # print(records_logger.handlers)
            records_logger.info(','.join([str(epoch)] + [str(round(x, 6)) for x in all_results]))
            # if save_path:
            #     self.save_model(save_path + '{}_{}.pkl'.format(self.model_name, epoch))

        # fw.close()
        return final_results

    def save_model(self, path):
        # torch.save(self, path)
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        print('Save successfully!')

    def load_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print('Load successfully!')
        return self

    def evaluate(self, dataloader, phase='val', record_name=None, graph=None, aux=False):
        self.eval()
        all_true_values = []
        all_predicted_values = []
        all_predicted_aux_values = []
        all_true_aux_values = []
        all_times = []
        all_ids = []
        all_loss = 0
        all_aux_loss = 0
        start_time = time.time()

        cur_graphs = None
        if graph:
            # load graphs_data
            self.node_trans = graph['node_trans']
            # self.graph = graph['data'].to(self.device)
            # self.graphs_data = graph['data']
            # cur_graphs = self.graphs_data[phase]
            cur_graphs = graph['data'][phase]
            cur_graphs = self.graph_to_device(cur_graphs)

        with torch.no_grad():
            for idx, (x, values, lengths, masks, ids, times, blocks) in enumerate(dataloader):
                loss = 0
                self.other_loss = 0
                # 转移数据
                if type(x) == list:
                    x = [content.to(self.device) for content in x]
                else:
                    x = x.to(self.device)
                # print('io-time:', time.time()-start_time)
                values = values.to(self.device)
                lengths = lengths.to(self.device)
                masks = masks.to(self.device)
                if blocks:
                    cur_graphs = blocks

                # if self.graphs_data:
                if graph:
                    ids = torch.tensor(list(map(lambda x: self.node_trans['paper'][x], ids)))
                    # cur_graphs = self.load_graph(self.graphs_data[phase])
                    # cur_graphs = self.graphs_data[phase]

                predicted_values, aux_output = self.predict(x, lengths, masks, ids, cur_graphs, times)
                if aux:
                    new_gt = self.get_new_gt(values)
                    # predicted_values, aux_output = predicted_values
                    aux_loss = self.get_aux_loss(aux_output, new_gt)
                    all_aux_loss = all_aux_loss + aux_loss.item()
                    self.other_loss = self.other_loss + aux_loss
                    predicted_aux_values = self.get_aux_pred(aux_output)

                    all_predicted_aux_values.append(predicted_aux_values.detach().cpu())
                    all_true_aux_values.append(new_gt.detach().cpu())
                if type(predicted_values) != torch.Tensor:
                    predicted_values = predicted_values[0]
                # print(predicted_values)
                true_values = values.unsqueeze(dim=-1)

                if self.relu_out:
                    predicted_values = self.relu_out(predicted_values)
                if dataloader.log:
                    loss = self.criterion(predicted_values, (true_values + 1).log())
                else:
                    loss = self.criterion(predicted_values, (true_values - dataloader.mean) / dataloader.std)

                if dataloader.log:
                    all_predicted_values.append(predicted_values.detach().cpu().exp() - 1)
                else:
                    all_predicted_values.append(predicted_values.detach().cpu() * dataloader.std + dataloader.mean)
                all_true_values.append(true_values.detach().cpu())
                # other info
                all_times.append(times)
                all_ids.extend(ids)

                all_loss += loss.item()

            elapsed = time.time() - start_time
            print('-' * 59)
            print('| end of {} | time: {:5.2f}s |'.format(phase, elapsed))
            logging.info('-' * 59)
            logging.info('| end of {} | time: {:5.2f}s |'.format(phase, elapsed))
            # all_predicted_values = torch.cat(all_predicted_values, dim=1).numpy()
            # all_true_values = torch.cat(all_true_values, dim=1).numpy()
            all_predicted_values = torch.cat(all_predicted_values, dim=0).numpy()
            all_true_values = torch.cat(all_true_values, dim=0).numpy()
            all_times = torch.cat(all_times, dim=0).squeeze(dim=-1).numpy()
            avg_loss = all_loss / len(dataloader)
            results = [avg_loss] + eval_result(all_true_values, all_predicted_values)
            format_str = result_format(results)
            for line in format_str.split('\n'):
                logging.info(line)

            if aux:
                print('=' * 89)
                logging.info('=' * 89)
                avg_aux_loss = all_aux_loss / len(dataloader)
                all_predicted_aux_values = torch.cat(all_predicted_aux_values, dim=0).numpy()
                all_true_aux_values = torch.cat(all_true_aux_values, dim=0).numpy()
                aux_results = [avg_aux_loss] + eval_aux_results(all_true_aux_values, all_predicted_aux_values)
                aux_format_str = aux_result_format(aux_results, phase=phase)

                for line in aux_format_str.split('\n'):
                    logging.info(line)

                print('=' * 89)
                logging.info('=' * 89)
                results += aux_results

            if self.record_path and (phase == 'test'):
                # print(all_times.shape, all_true_values.shape, all_predicted_values.shape)
                result_data = {
                    'time': all_times,
                    'true': np.squeeze(all_true_values, axis=-1),
                    'pred': np.squeeze(all_predicted_values, axis=-1)
                }
                if aux:
                    result_data['aux_pred'] = np.argmax(all_predicted_aux_values, axis=-1)
                    result_data['aux_true'] = all_true_aux_values

                df = pd.DataFrame(index=all_ids, data=result_data)
                if self.test_interval == 1:
                    record_name = record_name.format(self.model_name) if record_name \
                        else '{}_results_{}.csv'.format(self.model_name, self.epoch)
                else:
                    record_name = record_name.format(self.model_name) if record_name \
                        else '{}_results.csv'.format(self.model_name)
                # df.to_csv(self.record_path + '{}_results.csv'.format(self.model_name))
                df.to_csv(self.record_path + record_name)

            return results

    def test(self, test_dataloader, phase='test', record_name=None, graph=None, aux=False):
        results = self.evaluate(test_dataloader, phase, record_name=record_name, graph=graph, aux=aux)
        return results

    def show_results(self, dataloader, phase='val', record_name=None, graph=None, aux=False, return_graph=False, return_weight=False):
        print('===start showing===')
        self.eval()
        all_times = []
        all_ids = []
        all_outputs = []
        start_time = time.time()

        cur_graphs = None
        if graph:
            # load graphs_data
            self.node_trans = graph['node_trans']
            # self.graph = graph['data'].to(self.device)
            # self.graphs_data = graph['data']
            # cur_graphs = self.graphs_data[phase]
            cur_graphs = graph['data'][phase]
            cur_graphs = self.graph_to_device(cur_graphs)

        # print('===', dataloader)
        with torch.no_grad():
            for idx, (x, values, lengths, masks, ids, times, blocks) in enumerate(dataloader):
                # print(idx)
                loss = 0
                self.other_loss = 0
                # 转移数据
                if type(x) == list:
                    x = [content.to(self.device) for content in x]
                else:
                    x = x.to(self.device)
                # print('io-time:', time.time()-start_time)
                values = values.to(self.device)
                lengths = lengths.to(self.device)
                masks = masks.to(self.device)
                if blocks:
                    cur_graphs = blocks

                # if self.graphs_data:
                all_ids.extend(ids)
                all_times.extend(times.numpy().tolist())
                if graph:
                    ids = torch.tensor(list(map(lambda x: self.node_trans['paper'][x], ids)))
                    # cur_graphs = self.load_graph(self.graphs_data[phase])
                    # cur_graphs = self.graphs_data[phase]
                outputs = self.show(x, lengths, masks, ids, cur_graphs, times, return_graph=return_graph, return_weight=return_weight)
                # print(outputs)
                all_outputs.append(outputs)

        all_outputs = list(zip(*all_outputs))  # [k, batches]
        all_outputs_list = [[] for i in range(len(all_outputs))]
        for i in range(len(all_outputs_list)):
            for temp in all_outputs[i]:
                # print(type(temp))
                if type(temp) is list:
                    all_outputs_list[i].extend(temp)
                else:
                    all_outputs_list[i].append(temp)
            if type(all_outputs_list[i][0]) is np.ndarray:
                all_outputs_list[i] = np.concatenate(all_outputs_list[i], axis=0)
                print(all_outputs_list[i].shape)
            print(len(all_outputs_list[i]))
        # all_results = [all_ids, all_times] + all_outputs_list  # embs, graphs
        all_embs = [all_ids, all_times] + [all_outputs_list[0]]
        if return_weight:
            all_weights = [all_ids, all_times] + all_outputs_list[1:]
            return all_weights
        elif len(all_outputs_list) > 1:
            all_graphs = [all_ids, all_times] + [all_outputs_list[1]]
            return all_embs, all_graphs
        else:
            return all_embs

    def get_test_results(self, dataloader, save_path, record_path, graph=None, best_metric='male', aux=False):
        metrics = 'loss, acc, mae, r2, mse, rmse, Mrse, mrse, male, log_r2, msle, rmsle, smape, mape'.split(', ')
        if aux:
            metrics += 'aux_loss, acc, roc_auc, log_loss_value, ' \
                       'micro_prec, micro_recall, micro_f1, ' \
                       'macro_prec, macro_recall, macro_f1'.split(', ')
            if 'aux' not in self.model_name:
                self.model_name = self.model_name + '_aux'
                if self.aux_weight != 0.5:
                    self.model_name = self.model_name + '_aw{}'.format(self.aux_weight)
        raw_header = ',' + ','.join(['{}_' + metric for metric in metrics])
        global_best = self.model_name + '.pkl'
        ptr = re.compile('^' + self.model_name.replace('+', '\+') + '_\d.pkl$')
        ckpts = sorted([ckpt for ckpt in os.listdir(save_path) if ptr.match(ckpt)] + [global_best])
        print(ckpts)
        records_logger = logging.getLogger('records')
        self.record_path = record_path
        # records_logger.setLevel(logging.DEBUG)
        if self.seed:
            print('{}_records_{}_test.csv'.format(self.model_name, self.seed))
            fh = logging.FileHandler(record_path + '{}_records_{}_test.csv'.format(self.model_name, self.seed), mode='w+')

        else:
            fh = logging.FileHandler(record_path + '{}_records_test.csv'.format(self.model_name), mode='w+')
        records_logger.handlers = []
        records_logger.addHandler(fh)

        records_logger.warning('epoch'
                               + raw_header.replace('{}', 'test'))

        all_metric_values = []
        metric_index = metrics.index(best_metric)
        count = 0
        for ckpt in ckpts:
            try:
                self.load_model(save_path + ckpt)
                test_results = self.test(dataloader, graph=graph, record_name='{}_results_' + str(count) + '.csv', aux=aux)
                all_metric_values.append(test_results[metric_index])
                records_logger.info(','.join([ckpt] + [str(round(x, 6)) for x in test_results]))
                count += 1
            except Exception as e:
                print(e)

        if best_metric in ['acc', 'r2', 'log_r2']:
            best_ckpt_index = np.argmax(all_metric_values)
        else:
            best_ckpt_index = np.argmin(all_metric_values)

        print(best_ckpt_index)
        # self.load_model(save_path + ckpts[best_ckpt_index])
        # test_results = self.test(dataloader, graph=graph)

    def get_show_results(self, dataloader, save_path, record_path, graph=None, best_metric='male', aux=False, phase='test',
                        show='weight'):
        if show == 'weight':
            return_weight = True
            return_graph = False
        else:
            return_weight = False
            return_graph = True
        if aux:
            if 'aux' not in self.model_name:
                self.model_name = self.model_name + '_aux'
                if self.aux_weight != 0.5:
                    self.model_name = self.model_name + '_aw{}'.format(self.aux_weight)
        global_best = self.model_name + '.pkl'
        # records_logger = logging.getLogger('records')
        self.record_path = record_path
        # records_logger.setLevel(logging.DEBUG)
        print(global_best)
        self.load_model(save_path + global_best)
        all_show_results = self.show_results(dataloader, graph=graph, aux=aux, phase=phase, return_graph=return_graph,
                                             return_weight=return_weight)
        if return_weight:
            joblib.dump(all_show_results, save_path + self.model_name + '_weights.pkl')
        elif type(all_show_results) != list:
            joblib.dump(all_show_results[0], save_path + self.model_name + '_embs_s.pkl')
            joblib.dump(all_show_results[1], save_path + self.model_name + '_graphs.pkl')
        else:
            joblib.dump(all_show_results, save_path + self.model_name + '_embs.pkl')

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print('error! No such method!')
    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done base_model!')
