import time

import dgl
import torch

from layers.GCN.CL_layers import simple_cl
from models.base_model import BaseModel
from our_models.HCL import HCLWrapper

class FCLWrapper(HCLWrapper):
    def __init__(self, encoder: BaseModel, cl_type='simple', aug_type=None, tau=0.4, cl_weight=0.5, aug_rate=0.1, **kwargs):
        super(FCLWrapper, self).__init__(encoder, cl_type, aug_type, tau, cl_weight, aug_rate, **kwargs)
        self.model_name = self.model_name.replace('_HCL_', '_FCL_')
        if 'DCTSGCN' in self.model_name:
            self.model_name = 'H2CGL'

    def get_cl_inputs(self, snapshots, require_grad=True):
        if require_grad:
            readout = dgl.readout_nodes(snapshots, 'h', op=self.encoder.time_pooling, ntype='snapshot')
        else:
            readout = dgl.readout_nodes(snapshots, 'h', op=self.encoder.time_pooling, ntype='snapshot').detach()
        readout = self.proj_heads['snapshot'](readout)
        # print(readout.shape)
        return readout

    def cl(self, s1, s2, ids):
        # cl_loss_list = []
        cl_loss = 0
        if self.cl_type in ['simple', 'focus', 'cross_snapshot', 'focus_cross_snapshot']:
            temp_loss = simple_cl(s1, s2, self.tau)
            cl_loss = cl_loss + temp_loss
        elif self.cl_type.endswith('hard_negative'):
            sample_num = self.encoder.hn
            true_len = s1.shape[0] // sample_num if sample_num else s1.shape[0]
            temp_index_list = []
            for i in range(sample_num):
                temp_index_list.append(range(true_len * i, true_len * (i + 1)))
            # cur_index = list(zip(range(true_len), range(true_len, true_len * 2)))
            cur_index = list(zip(*temp_index_list))
            s1 = s1[cur_index, :]
            s2 = s2[cur_index, :]
            # print(s1.shape)
            temp_loss = simple_cl(s1, s2, self.tau)
            cl_loss = cl_loss + temp_loss
        return cl_loss

    def forward(self, content, lengths, masks, ids, graph, times, **kwargs):
        # cur_time = time.time()
        # g1_snapshots, g2_snapshots, cur_snapshot_adj = self.get_graphs(ids, graph)
        # g1_inputs = self.encoder.get_graphs(ids, graph, aug=(self.aug_methods[0], self.aug_configs[0]))
        # g2_inputs = self.encoder.get_graphs(ids, graph, aug=(self.aug_methods[1], self.aug_configs[1]))
        if self.cl_type.endswith('aug_hard_negative'):
            all_inputs = self.encoder.get_graphs(content, lengths, masks, ids, graph, times,
                                                           phase='train_combined', df_hn=self.df_hn)
        else:
            all_inputs = self.encoder.get_graphs(content, lengths, masks, ids, graph, times,
                                                           phase='train_combined')
        # print('get graphs done:{:.3f}s'.format(time.time() - cur_time))
        # cur_time = time.time()
        all_out, all_snapshots = self.encoder.encode(all_inputs)
        # print('encode done:{:.3f}s'.format(time.time() - cur_time))
        all_snapshots = self.get_cl_inputs(all_snapshots)

        batch_size = all_out.shape[0] // 2
        # print(batch_size, all_out.shape[0])
        g1_out, g2_out = all_out[:batch_size, :], all_out[batch_size:, :]
        g1_snapshots, g2_snapshots = all_snapshots[:batch_size, :], all_snapshots[batch_size:, :]

        # cur_time = time.time()
        self.other_loss = self.cl_weight * self.cl(g1_snapshots, g2_snapshots, ids)
        # logging.info('cl done:{:.3f}s'.format(time.time() - cur_time))
        # print('cl done:{:.3f}s'.format(time.time() - cur_time))

        time_out = (g1_out + g2_out) / 2
        output = self.encoder.decode(time_out)
        # print(time_out.shape)
        # logging.info('=' * 20 + 'finish' + '=' * 20)
        return output
