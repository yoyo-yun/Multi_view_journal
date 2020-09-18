import torch
import torch.nn as nn
from models.layers.token_emb import TokenEmbedding
from transformers import DistilBertModel, BertModel


class GloveLSTMEncoder(nn.Module):
    def __init__(self, config):
        super(GloveLSTMEncoder, self).__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.embedding.size(0), config.embedding.size(1), pad_idx=config.pad_idx)
        self.embedding.weight.data.copy_(config.embedding)
        self.embedding.weight.requires_grad = False
        self.contextualized_encoder = nn.LSTM(config.words_dim, config.contextual_dim // 2, batch_first=True,
                                           bidirectional=True)

    def forward(self, x):
        hidden_state, _ = self.contextualized_encoder(self.embedding(x))
        return hidden_state


class GloveEncoder(nn.Module):
    def __init__(self, config):
        super(GloveEncoder, self).__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.embedding.size(0), config.embedding.size(1), pad_idx=config.pad_idx)
        self.embedding.weight.data.copy_(config.embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        # self.pretrained_weights = 'distilbert-base-uncased'
        # self.encoder = DistilBertModel.from_pretrained(self.pretrained_weights, output_hidden_states=True)

        self.pretrained_weights = 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained(self.pretrained_weights, output_hidden_states=True)

    def forward(self, x, mask=None):
        # x (batch, sent_size)
        # return self.encoder(x, attention_mask=mask)[-1]  # all hidden_states: tuple(13, batch, max_length, 768)
        return self.encoder(x, attention_mask=mask)[-1][-2]


class RefineBERT(nn.Module):
    def __init__(self, config):
        super(RefineBERT, self).__init__()
        # self.pretrained_weights = 'distilbert-base-uncased'
        # self.encoder = DistilBertModel.from_pretrained(self.pretrained_weights, output_hidden_states=True)

        self.pretrained_weights = 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained(self.pretrained_weights, output_hidden_states=True).train()
        self.classifier = nn.Linear(768, config.num_classes)

    def forward(self, x, mask=None):
        # x (batch, sent_size)
        return self.classifier(self.encoder(x, attention_mask=mask)[1])


class BERTLSTMEncoder(nn.Module):
    def __init__(self, config):
        super(BERTLSTMEncoder, self).__init__()
        self.pretrained_weights = 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained(self.pretrained_weights, output_hidden_states=True)
        self.contextualized_encoder = nn.LSTM(config.words_dim, config.contextual_dim // 2, batch_first=True,
                                           bidirectional=True)
    def forward(self, x , mask=None):
        encoder = self.encoder(x, attention_mask=mask)[-1] # tuple(12, batch, max_length, 768)
        encoder = encoder[-2]  # (batch, max_length, 768)
        hidden_state, _ = self.contextualized_encoder(encoder)
        return hidden_state