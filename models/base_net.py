import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from models.layers.blocks import MVBlock


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_views = config.n_views
        self.views = Parameter(torch.Tensor(self.n_views, self.config.view_dim).uniform_(-0.2, 0.2))
        self.multi_view_block = MVBlock(config)
        self.pre_classifier = nn.Linear(config.sent_hidden_dim * config.n_views, config.pre_classifier_dim)
        self.classifier = nn.Linear(config.pre_classifier_dim, config.num_classes)

    def forward(self, hidden_state, mask=None):
        batch_size = hidden_state.size(0)
        view = self.views.unsqueeze(1).repeat(1, batch_size, 1)  # (n_view, bs, dim)
        all_sent_repr, all_att, all_att_logits = self.multi_view_block(hidden_state, view, mask=mask)
        all_sent_repr = torch.cat(all_sent_repr, -1)
        pre_logits = torch.tanh(self.pre_classifier(all_sent_repr))
        logits = self.classifier(pre_logits)

        return logits, [view.transpose(0, 1)], all_att, all_att_logits


class MulNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_views = config.n_views
        self.views = Parameter(torch.Tensor(self.n_views, self.config.view_dim).uniform_(-0.2, 0.2))
        self.multi_view_block = MVBlock(config)
        # self.pre_classifier = nn.ModuleList([nn.Linear(config.sent_hidden_dim, config.pre_classifier_dim) for _ in range(self.n_views)])
        self.classifier = nn.ModuleList([nn.Linear(config.sent_hidden_dim, 1) for _ in range(self.n_views)])

    def forward(self, hidden_state, mask=None):
        batch_size = hidden_state.size(0)
        view = self.views.unsqueeze(1).repeat(1, batch_size, 1)  # (n_view, bs, dim)
        all_sent_repr, all_att, all_att_logits = self.multi_view_block(hidden_state, view, mask=mask)

        logits = [classifier(feature) for feature, classifier in
                  zip(all_sent_repr, self.classifier)]
        logits = torch.cat(logits, dim=-1)

        return logits, [view.transpose(0, 1)], all_att, all_att_logits


class MulNetV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_views = config.n_views
        self.views = Parameter(torch.Tensor(self.n_views, self.config.view_dim).uniform_(-0.2, 0.2))
        self.multi_view_block = MVBlock(config)
        self.pre_classifier = nn.Linear(config.sent_hidden_dim * config.n_views, config.pre_classifier_dim)
        self.classifier = nn.Linear(config.pre_classifier_dim, config.num_classes)

    def forward(self, hidden_state, mask=None):
        batch_size = hidden_state.size(0)
        view = self.views.unsqueeze(1).repeat(1, batch_size, 1)  # (n_view, bs, dim)
        all_sent_repr, all_att, all_att_logits = self.multi_view_block(hidden_state, view, mask=mask)

        all_sent_repr = torch.cat(all_sent_repr, -1)
        pre_logits = torch.tanh(self.pre_classifier(all_sent_repr))
        logits = self.classifier(pre_logits)

        return logits, [view.transpose(0, 1)], all_att, all_att_logits
