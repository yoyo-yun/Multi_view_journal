# multi-view-block

import math
import torch
import torch.nn as nn

import torch.nn.init as init
from torch.nn.parameter import Parameter


class Bilinear(nn.Module):
    def __init__(self, config, drop=0.):
        super().__init__()
        self.features1 = config.view_dim
        self.features2 = config.contextual_dim
        self.bilinear_size = config.bilinear_size
        self.drop = nn.Dropout(p=drop)
        self.weight = Parameter(torch.Tensor(self.bilinear_size, self.features1, self.features2))
        self.reset_parameters()
        self.bilinear = nn.Bilinear(self.features1, self.features2, self.bilinear_size)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, u, h):
        # self.u = u
        # batch_size = h.size(0)
        # out = u @ self.weight @ (h.contiguous().view(-1, self.features2).transpose(-1, -2))
        # out = out.squeeze().transpose(-1, -2).contiguous().view(batch_size, -1, self.bilinear_size)
        # u: (1,dim1) h:(batch, seq, dim2)
        # out = self.bilinear(u.unsqueeze(0).contiguous().repeat(h.size(0), h.size(1), 1), h)

        # u: (n_view, dim1) h: (batch, seq, dim2) w: (k, dim1, dim2)
        # x = torch.einsum('ij, kjl->ikl', u, self.weight)  # (n_view, k, dim2)
        # out = torch.einsum('ijk, bsk->ibsj', x, h)  # (n_view, batch, seq, k)

        # u: (n_views, bs, dim1) w: (k, dim1, dim2) h: (batch, seq, dim2)
        x = torch.einsum('ibj,kjl->iblk', u, self.weight)  # (n_view, bs, dim2, k)
        out = torch.einsum('ijkl,jbk->ijbl', x, h)  # (n_view, bs, seq, k)

        return self.drop(out)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, config):
        super(SelfAttention, self).__init__()
        # self.pre_pooling_linear = nn.Linear(input_dim, config.pre_pooling_dim)
        # self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)
        self.pooling_linear = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        # input: text representation
        # weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear(x))).squeeze(dim=2)
        weights = self.pooling_linear(x).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        att_score = nn.Softmax(dim=-1)(weights)

        # return torch.mul(x, weights.unsqueeze(2)).sum(dim=1)
        return att_score.unsqueeze(2), weights.unsqueeze(2)


class Fusion(nn.Module):
    def __init__(self, feature_dim1, feature_dim2, mode='gate'):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(feature_dim1, 1)
        self.linear2 = nn.Linear(feature_dim2, 1)
        self.linear3 = nn.Linear(feature_dim1 + feature_dim2, feature_dim1)
        self.mode = mode

    def forward(self, feature1, feature2):
        if self.mode == 'gate':
            assert feature1.shape == feature2.shape, "same dimension of both feature is need"
            return self.gate(feature1, feature2)
        elif self.mode == 'concate':
            return self.concate(feature1, feature2)
        elif self.mode == 'add':
            assert feature1.shape == feature2.shape, "same dimension of both feature is need"
            return self.add(feature1, feature2)
        elif self.mode == 'deep_fusion':
            return self.deep_fusion(feature1, feature2)
        elif self.mode == 'first':
            return feature1
        elif self.mode == 'second':
            return feature2
        else:
            raise RuntimeError("unsupported fusion mode accured")

    def gate(self, feature1, feature2):
        gate_sore = torch.sigmoid(self.linear1(feature1) + self.linear2(feature2))
        return gate_sore * feature1 + (1 - gate_sore) * feature2

    def concate(self, feature1, feature2):
        return torch.cat([feature1, feature2], -1)

    def add(self, feature1, feature2):
        return torch.add(feature1, feature2)

    def deep_fusion(self, feature1, feature2):
        return self.linear3(self.concate(feature1, feature2))


class SubBlock(nn.Module):
    '''
    :argument
        input: 1. hidden state [batch, seq, dim] 2. tensor [1,dim]
        output: (gated_feature, attention) where gated_feature: [batch, dim], attention(batch, seq)
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru = nn.LSTM(config.bilinear_size, config.contextual_dim // 2, batch_first=True, bidirectional=True)
        self.att = SelfAttention(config.contextual_dim, config)
        self.fusion = Fusion(config.contextual_dim, config.contextual_dim, mode='gate')

    def forward(self, hidden_state, complex_repr, mask=None):
        gru_repr, _ = self.gru(complex_repr)
        att_map, att_logits = self.att(gru_repr, mask=mask)  # shape: [batch, seq, 1]
        hidden_sent_repr = torch.mul(hidden_state, att_map).sum(1)  # [batch, dim1]
        gru_sent_repr = torch.mul(gru_repr, att_map).sum(1)  # [batch, dim2]
        sent_repre = self.fusion(hidden_sent_repr, gru_sent_repr)
        return sent_repre, att_map, att_logits


class MVBlock(nn.Module):
    def __init__(self, config):
        super(MVBlock, self).__init__()
        self.config = config
        self.bilinear = Bilinear(config, drop=0.)
        self.blocks = nn.ModuleList([SubBlock(config) for _ in range(config.n_views)])

    def forward(self, hidden_state, tensor, mask=None):
        complex_reprs = self.bilinear(tensor, hidden_state)  # (n_view, batch, seq, k)
        all_sent_repres = []
        all_atts = []
        all_atts_logits = []
        for i, block in enumerate(self.blocks):
            sent_repres, atts, atts_logits = block(hidden_state, complex_reprs[i], mask=mask)
            all_sent_repres.append(sent_repres)
            all_atts.append(atts.squeeze(-1))
            all_atts_logits.append(atts_logits.squeeze(-1))
        return all_sent_repres, all_atts, all_atts_logits  # [(batch, dim),...,(batch, dim)], [(batch, seq),...,(batch, seq)]
