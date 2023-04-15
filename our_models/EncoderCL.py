import copy

import pandas as pd
import torch

from models.base_model import BaseModel
from our_models.CL import CLWrapper

class EncoderCLWrapper(CLWrapper):
    def __init__(self, encoder: BaseModel, cl_type='simple'):
        super(EncoderCLWrapper, self).__init__(encoder, cl_type, None)
        # self.aux_criterion = encoder.aux_criterion
        self.vice_encoder = copy.deepcopy(self.encoder)
        self.eta = 1.
        self.model_name = self.encoder.model_name + '_ECL_' + cl_type

    def load_graphs(self, graphs):
        # no graph aug
        dealt_graphs = self.encoder.load_graphs(graphs)
        df_hn = pd.read_csv('./data/{}/hard_negative.csv'.format(graphs['data_source']), index_col=0)
        paper_trans = graphs['node_trans']['paper']
        df_hn.index = [paper_trans[str(paper)] for paper in df_hn.index]
        for column in df_hn:
            df_hn[column] = df_hn[column].apply(lambda x: [paper_trans[str(paper)] for paper in eval(x)])
        self.df_hn = df_hn
        return dealt_graphs

    def get_aug_encoder(self):
        for (adv_name, adv_param), (name, param) in zip(self.vice_encoder.named_parameters(),
                                                        self.encoder.named_parameters()):
            if (not name.startswith('fc')) | ('text' not in name):
                # print(name, param.data.std())
                adv_param.data = param.data.detach() + self.eta * torch.normal(0, torch.ones_like(
                    param.data) * param.data.std()).detach().to(self.device)

    def forward(self, content, lengths, masks, ids, graph, times, **kwargs):
        # cur_inputs = self.encoder.get_graphs(ids, times, graph)
        if self.cl_type == 'aug_hard_negative':
            cur_inputs = self.encoder.get_graphs(content, lengths, masks, ids, graph, times, df_hn=self.df_hn)
        else:
            cur_inputs = self.encoder.get_graphs(content, lengths, masks, ids, graph, times)
        aug_inputs = copy.deepcopy(cur_inputs)
        self.get_aug_encoder()
        g1_out, g1_snapshots = self.encoder.encode(cur_inputs)
        g2_out, g2_snapshots = self.vice_encoder.encode(aug_inputs)
        # logging.info('encode done:{:.3f}s'.format(time.time() - cur_time))

        # cur_time = time.time()
        g1_snapshots = self.get_cl_inputs(g1_snapshots)
        g2_snapshots = self.get_cl_inputs(g2_snapshots, require_grad=False)
        self.other_loss = self.cl_weight * self.cl(g1_snapshots, g2_snapshots, ids)
        # logging.info('cl done:{:.3f}s'.format(time.time() - cur_time))

        time_out = g1_out
        output = self.encoder.decode(time_out)
        # print(time_out.shape)
        # logging.info('=' * 20 + 'finish' + '=' * 20)
        return output

