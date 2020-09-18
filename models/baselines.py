import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_size = config.sent_hidden_dim
        self.lstm = nn.LSTM(config.contextual_dim, config.sent_hidden_dim, batch_first=True,
                            bidirectional=False)
        self.pre_classifier = nn.Linear(config.sent_hidden_dim, config.pre_classifier_dim)
        self.classifier = nn.Linear(config.pre_classifier_dim, config.num_classes)

    def forward(self, x, mask=None):
        hidden_state, _ = self.lstm(x)
        if mask is not None:
            # features : (batch, time_step, dim)
            feature = hidden_state.gather(1, torch.relu(mask.long().sum(1, keepdim=True).repeat(1, self.feature_size).unsqueeze(
                1) - 1)).squeeze(1)
        else:
            feature = hidden_state[:, -1]
        logits = self.classifier(torch.tanh(self.pre_classifier(feature)))
        return logits, [], [], []


ConvMethod = "in_channel__is_embedding_dim"


class CNN(nn.Module):
    def __init__(self, config, kernel_sizes=[3, 4, 5], num_filters=100, embedding_dim=300, pretrained_embeddings=None):
        super(CNN, self).__init__()
        self.config = config
        self.kernel_sizes = kernel_sizes

        self.embedding.weight.data.copy_(pretrained_embeddings)

        conv_blocks = []
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size = sentence_len - kernel_size + 1

            if ConvMethod == "in_channel__is_embedding_dim":
                conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size,
                                   stride=1)
            else:
                conv1d = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size * embedding_dim,
                                   stride=embedding_dim)

            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size)
            )
            if use_cuda:
                component = component.cuda()

            conv_blocks.append(component)

            if 0:
                conv_blocks.append(
                    nn.Sequential(
                        conv1d,
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=maxpool_kernel_size)
                    ).cuda()
                )

        self.conv_blocks = nn.ModuleList(conv_blocks)  # ModuleList is needed for registering parameters in conv_blocks
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):  # x: (batch, sentence_len)
        if ConvMethod == "in_channel__is_embedding_dim":
            #    input:  (batch, in_channel=1, in_length=sentence_len*embedding_dim),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            x = x.transpose(1, 2)  # needs to convert x to (batch, embedding_dim, sentence_len)
        else:
            #    input:  (batch, in_channel=embedding_dim, in_length=sentence_len),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            x = x.view(x.size(0), 1, -1)  # needs to convert x to (batch, 1, sentence_len*embedding_dim)

        x_list = [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        feature_extracted = out
        out = F.dropout(out, p=0.5, training=self.training)
        # return F.softmax(self.fc(out), dim=1), feature_extracted
        return self.fc(out), None, None, None


from torch.autograd import Variable


class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """

    def __init__(self, batch_size, lstm_hid_dim, d_a, r, max_len, emb_dim=300, vocab_size=None,
                 use_pretrained_embeddings=False, embeddings=None, type=0, n_classes=1):
        """
        Initializes parameters suggested in paper

        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes

        Returns:
            self

        Raises:
            Exception
        """
        super(StructuredSelfAttention, self).__init__()

        self.embeddings, emb_dim = self._load_embeddings(use_pretrained_embeddings, embeddings, vocab_size, emb_dim)
        self.lstm = torch.nn.LSTM(emb_dim, lstm_hid_dim, 1, batch_first=True)
        self.linear_first = torch.nn.Linear(lstm_hid_dim, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(lstm_hid_dim, self.n_classes)
        self.batch_size = batch_size
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.hidden_state = self.init_hidden()
        self.r = r
        self.type = type

    def _load_embeddings(self, use_pretrained_embeddings, embeddings, vocab_size, emb_dim):
        """Load the embeddings based on flag"""

        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")

        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")

        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=1)

        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1), padding_idx=1)
            # word_embeddings.weight = torch.nn.Parameter(embeddings)
            word_embeddings.weight = torch.nn.Parameter(embeddings, requires_grad=False)
            emb_dim = embeddings.size(1)

        return word_embeddings, emb_dim

    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n

        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors


        """

        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def init_hidden(self):
        return (Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim).cuda()),
                Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim).cuda()))

    def forward(self, x):
        embeddings = self.embeddings(x)
        outputs, self.hidden_state = self.lstm(embeddings.view(self.batch_size, self.max_len, -1), self.hidden_state)
        x = F.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        sentence_embeddings = attention @ outputs
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r

        if not bool(self.type):
            output = F.sigmoid(self.linear_final(avg_sentence_embeddings))

            return output, attention
        else:
            # return F.log_softmax(self.linear_final(avg_sentence_embeddings)), attention
            # return F.sigmoid(self.linear_final(avg_sentence_embeddings)), attention
            return self.linear_final(avg_sentence_embeddings), attention

            # Regularization

    def l2_matrix_norm(self, m):
        """
        Frobenius norm calculation

        Args:
           m: {Variable} ||AAT - I||

        Returns:
            regularized value


        """
        return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5).type(torch.DoubleTensor)
