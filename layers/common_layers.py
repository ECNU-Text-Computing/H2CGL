import torch
import torch.nn as nn
import math


class PositionalEncodingSC(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncodingSC, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: Tensor, shape [seq_len, batch_size, embedding_dim]
        :return Tensor:
        """
        x = x + self.pe[:x.size(0)]
        out = self.dropout(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.squeeze(dim=1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: Tensor, shape [batch_size, seq_len]
        :return Tensor:
        """
        # x = x + self.pe[:x.size(0)]
        # out = self.dropout(x)
        out = self.pe[x]
        return out


class TemporalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000, random=True):
        super(TemporalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # t
        div_term = 1 / math.sqrt(d_model)
        if random:
            w = torch.randn((1, d_model))
        else:
            w = torch.arange(0, d_model, 1).unsqueeze(0) / d_model
        # temp = nn.Parameter(position * w)
        # print(position.shape, w.shape)
        temp = position * w
        te = torch.cos(temp) * torch.cos(temp) * div_term
        # print(te)
        self.register_buffer('te', te)

    def forward(self, x):
        """
        :param x: Tensor, shape [batch_size, seq_len]
        :return Tensor:
        """
        # print(self.te.device)
        out = self.te[x]
        return out


class AttentionModule(nn.Module):
    def __init__(self, dim_in, dim_out=None, activation=nn.Tanh()):
        super(AttentionModule, self).__init__()
        if dim_out is None:
            dim_out = dim_in
        # self.trans = nn.Linear(dim_in, dim_out)
        # self.seq_trans = nn.Linear(dim_in, dim_out)
        self.target_trans = MLP(dim_in, dim_out)
        self.seq_trans = MLP(dim_in, dim_out)
        # self.w = nn.Linear(dim_out, 1, bias=False)
        self.w = nn.Linear(dim_out, dim_out, bias=False)
        self.u = nn.Linear(dim_out, dim_out, bias=False)
        self.activation = activation

    def forward(self, target, sequence, mask):
        # batch_size = target.shape[0]
        target = self.target_trans(target)
        # print(sequence.shape)
        sequence = self.seq_trans(sequence)

        # attn = self.activation(sequence @ self.w(target).unsqueeze(dim=-1)) + mask.unsqueeze(dim=-1).detach()
        # print((self.u(sequence) @ self.w(target).unsqueeze(dim=-1)).shape)
        # attn = self.activation(self.u(sequence) @ self.w(target).unsqueeze(dim=-1)) + mask.unsqueeze(dim=-1).detach()
        attn = self.u(sequence) @ self.w(target).unsqueeze(dim=-1) + mask.unsqueeze(dim=-1).detach()
        # attn = attn.view(batch_size, -1)
        # attn[mask] = -1e8
        attn = torch.softmax(attn, dim=-2)

        out = (sequence * attn).sum(1)
        # print(out.shape)
        return out


class GateModule(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(GateModule, self).__init__()
        self.input_trans = MLP(input_dim, input_dim)
        self.target_trans = MLP(input_dim, input_dim)
        self.mix_trans = MLP(input_dim * 2, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_emb, target_emb):
        alpha = self.sigmoid(self.input_trans(input_emb) + self.target_trans(target_emb))
        input_emb = alpha * input_emb
        mix_out = self.mix_trans(torch.cat((input_emb, target_emb), dim=-1))
        return mix_out


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, dim_inner=None,
                 num_layers=2, activation=nn.ReLU(inplace=False), **kwargs):
        super(MLP, self).__init__()
        layers = []
        dim_inner = dim_in if dim_inner is None else dim_inner
        if num_layers > 1:
            for i in range(num_layers - 1):
                # print(dim_in, dim_inner, bias)
                layers.append(nn.Linear(dim_in, dim_inner, bias))
                if activation:
                    layers.append(activation)
            layers.append(nn.Linear(dim_inner, dim_out, bias))
        else:
            layers.append(nn.Linear(dim_in, dim_out, bias))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        batch = self.model(batch)
        return batch


class CapsuleNetwork(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, num_routing=3, num_caps=4, return_prob=False,
                 activation=nn.ReLU(), **kwargs):
        # projection and routing
        super(CapsuleNetwork, self).__init__()
        dim_capsule = dim_out // num_caps
        self.dim_out = dim_out
        self.model = OriginCaps(in_dim=dim_in, in_caps=1, num_caps=num_caps, dim_caps=dim_capsule,
                                num_routing=num_routing)
        self.bn_out = nn.BatchNorm1d(dim_capsule)  # bn is important for capsule
        self.fc_out = MLP(dim_out, dim_out, bias=bias, activation=activation)
        self.return_prob = return_prob

    def forward(self, x):
        # [N, dim_in]
        # x = self.bn_in(x)
        x = x.unsqueeze(dim=1)
        x, p = self.model(x)  # [N, Nc, dc] p是capsule probability
        # print(p)
        x = self.bn_out(x.transpose(1, 2)).transpose(1, 2)
        x = self.fc_out(x.reshape(-1, self.dim_out))
        # x = self.bn_out(x)
        if self.return_prob:
            return x, p
        else:
            return x


def squash(x, dim=-1):
    # squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    squared_norm = torch.square(x).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    # vector = scale * x / (squared_norm.sqrt() + 1e-8)
    vector = scale * x / torch.sqrt(squared_norm + 1e-11)
    return vector, scale


class OriginCaps(nn.Module):
    """Original capsule graph_encoder."""

    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing):
        """
        Initialize the graph_encoder.

        Args:
            in_dim: 		Dimensionality (i.e. length) of each capsule vector.
            in_caps: 		Number of input capsules if digits graph_encoder.
            num_caps: 		Number of capsules in the capsule graph_encoder
            dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(OriginCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        # self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim),
        #                       requires_grad=True)
        self.W = nn.Parameter(torch.empty(1, num_caps, in_caps, dim_caps, in_dim),
                              requires_grad=True)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.W, -0.1, 0.1)

    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        # print('x', torch.isnan(x).sum())
        #
        # W @ x =
        # (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, num_caps, in_caps, dim_caps, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, num_caps, in_caps, dim_caps)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1, requires_grad=False).to(x.device)

        for route_iter in range(self.num_routing - 1):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v, s = squash(s)
            # print('v', torch.isnan(v).sum())
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b = b + uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        # print(c.shape, u_hat.shape)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v, p = squash(s)
        # print('last_v', torch.isnan(x).sum(), torch.isnan(v).sum(), torch.isnan(c).sum(), torch.isnan(s).sum())
        # print(x.squeeze(1))
        # print(v)
        # if torch.isnan(v).sum() > 0:
        #     raise Exception
        # print(s)

        return v, p

if __name__ == '__main__':
    pe = PositionalEncoding(128)
    print(pe.pe.shape)
    print(pe.pe[[1, 1, 2]])
    x = torch.randint(10, (3, 5))
    print(pe(x).shape)
    # x = torch.tensor([1, 5, 10, 4, 2, 10])
    # te = TemporalEncoding(128)
    # print(te(x).shape)
    # print([torch.norm(te(x)[i]) for i in range(x.shape[0])])
    # te = TemporalEncoding(128, random=False)
    # print(te(x).shape)
    # print([torch.norm(te(x)[i]) for i in range(x.shape[0])])

    # model = OriginCaps(in_dim=256, in_caps=1, num_caps=4, dim_caps=64, num_routing=3)
    # # (batch_size, in_caps, in_dim)
    # x = torch.randn(10, 1, 256)
    # print(model(x)[0].shape)
    # # print(model(x).norm(2))
    # cn = CapsuleNetwork(dim_in=256, dim_out=256)
    # x = torch.randn(8, 256)
    # print(cn(x))
