import os
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
from datasets.dataset_from_Bert import Data
from models.get_optim import get_Adam_optim
from transformers import DistilBertTokenizer, BertTokenizer
from cfgs.constants import DATASET_TORCHTXT, MODEL, DATASET_BERT, ENCODER, MUL_MODEL

# pretrained_weights = 'distilbert-base-uncased'
# tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights)

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


def load_vectors(config, is_statistic=True):
    import time
    start_time = time.time()
    print("====loading vectors...")
    processor = DATASET_TORCHTXT[config.dataset](config.TRAIN.batch_size)
    train = processor.get_train_examples()
    dev = processor.get_dev_examples()
    test = processor.get_test_examples()
    # loading samples
    sample = processor.get_sample()
    print("done!")
    end_time = time.time()
    print("taking times: {:.2f}s.".format(end_time - start_time))
    print()

    config.num_classes = processor.NUM_CLASSES
    config.is_multilabel = processor.IS_MULTILABEL
    config.embedding = processor.pretrained_embedding
    config.pad_idx = processor.pad_idx

    print("===Train size       : " + str(len(train.dataset)))
    print()
    print("===Validation size  : " + str(len(dev.dataset)))
    print()
    print("===Test size        : " + str(len(test.dataset)))
    print()
    print("===common datasets information...")
    print("num_labels          : " + str(config.num_classes))
    print("pad_idx             : " + str(config.pad_idx))
    print("Vocabulary size     : " + str(config.embedding.shape[0]))
    if is_statistic: print("Basic statistics    : "); statics(train, dev, test)
    print()

    return train, dev, test, sample


def load_reuters(config):
    import h5py
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    with h5py.File(os.path.join('datasets', 'reuters', 'reuters_multilabel_dataset.hdf5'), "r") as f:
        embed_w2v = list(f["w2v"])
        train = list(f['train'])
        train_label = list(f['train_label'])
        test = list(f['test'])
        test_label = list(f['test_label'])

        for i, v in enumerate(train):
            if np.sum(v) == 0:
                del (train[i])
                del (train_label[i])

        for i, v in enumerate(test):
            if np.sum(v) == 0:
                del (test[i])
                del (test_label[i])

        train, dev, train_label, dev_label = train_test_split(train, train_label, test_size=0.1)

        train = np.array(train)
        dev = np.array(dev)
        test = np.array(test)
        train_label = np.array(train_label)
        dev_label = np.array(dev_label)
        test_label = np.array(test_label)

        config.pad_idx = 0
        config.num_classes = len(test_label[0])
        config.embedding = torch.from_numpy(np.array(embed_w2v))
        config.is_multilabel = True


        print("===Train size       : " + str(len(train)))
        print()
        print("===Validation size  : " + str(len(dev)))
        print()
        print("===Test size        : " + str(len(test)))
        print()
        print("===common datasets information...")
        print("num_labels          : " + str(config.num_classes))
        print("pad_idx             : " + str(config.pad_idx))
        print("Vocabulary size     : " + str(config.embedding.shape[0]))
        print("Basic statistics    : ")
        statics_reuster(train, dev, test)

    train = TensorDataset(torch.from_numpy(train).to(config.device), torch.from_numpy(train_label).to(config.device))
    dev = TensorDataset(torch.from_numpy(dev).to(config.device), torch.from_numpy(dev_label).to(config.device))
    test = TensorDataset(torch.from_numpy(test).to(config.device), torch.from_numpy(test_label).to(config.device))

    train_iterator = DataLoader(dataset=train,
                                batch_size=config.TRAIN.batch_size,
                                shuffle=True)
    dev_iterator = DataLoader(dataset=dev,
                              batch_size=config.TRAIN.batch_size,
                              shuffle=False)
    test_iterator = DataLoader(dataset=test,
                               batch_size=config.TRAIN.batch_size,
                               shuffle=False
                               )
    return train_iterator, dev_iterator, test_iterator, None


def statics_reuster(*datasets):
    lengths = []
    for dataset in datasets:
        for sample in dataset:
            lengths.append((sample != 0).sum())
    lengths = np.array(lengths)
    print(
        '''
        max_length is {},
        min_length is {},
        avg_length is {:.2f}
        '''.format(lengths.max(), lengths.min(), lengths.mean())
    )


def statics(*data_iteraters):
    lengths = []
    for data_iterater in data_iteraters:
        for batch in data_iterater:
            text, length = batch.text
            lengths.extend(length.tolist())
    lengths = np.array(lengths)
    print(
        '''
        max_length is {},
        min_length is {},
        avg_length is {:.2f}
        '''.format(lengths.max(), lengths.min(), lengths.mean())
    )


def load_net(config):
    if config.is_multilabel:
        config.n_views = config.num_classes
        # config.n_views = 1
        models = MUL_MODEL
    else:
        models = MODEL
    encoder = ENCODER[config.encoder](config).to(config.device)
    net = models[config.model](config).to(config.device)
    optim = get_Adam_optim(config, net, encoder)

    if config.n_gpu > 1:
        net = torch.nn.DataParallel(net)
        encoder = torch.nn.DataParallel(encoder)
    else:
        net = net
        encoder = encoder
    return encoder, net, optim


def multi_acc(y, preds):
    preds = torch.argmax(F.softmax(preds), dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def margin_loss(preds, y):
    y = y.float()
    # preds = torch.sigmoid(preds)
    loss = y * (torch.where(0 > 0.9 - preds, torch.zeros_like(preds), 0.9 - preds) ** 2) + \
           0.25 * (1.0 - y) * ((torch.where(0 > preds - 0.1, torch.zeros_like(preds), preds - 0.1)) ** 2)
    loss = torch.mean(torch.sum(loss, dim=-1))
    return loss


def multi_label_metric(logits, labels, threshold=0.4):
    # logits (bs, multi-lables) label (bs, multi-labels)
    logits = np.array(logits)
    preds_probs = np.array(logits)
    preds_probs[np.where(preds_probs >= threshold)] = 1.0
    preds_probs[np.where(preds_probs < threshold)] = 0.0

    labels = np.array(labels)

    [precision, recall, F1, support] = \
        precision_recall_fscore_support(labels, preds_probs, average='micro')
    acc = accuracy_score(labels, preds_probs)
    # print(labels[:10])
    # print(preds_probs[:10])
    # print('\rER: %.3f' % acc, 'Precision: %.3f' % precision, 'Recall: %.3f' % recall, 'F1: %.3f' % F1)
    return acc, precision, recall, F1


def bertTokenizer(batch_example, config):
    text, label = batch_example
    t = tokenizer.batch_encode_plus(text, max_length=int(config.max_seq_length),
                                    pad_to_max_length=True)
    input_ids = torch.tensor(t["input_ids"])
    input_mask = t["attention_mask"]
    return input_ids.to(config.device), torch.tensor(input_mask).to(config.device), label.to(config.device)


def load_bert_dataset(config):
    processor = DATASET_BERT[config.dataset]()
    config.max_seq_length = processor.MAX_SEQ_LENGTH
    config.num_classes = processor.NUM_CLASSES
    config.is_multilabel = processor.IS_MULTILABEL

    train_examples = processor.get_train_examples()
    train_texts = [example.text for example in train_examples]
    train_labels = [example.label for example in train_examples]
    train_dataset = Data(train_texts, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.batch_size, shuffle=True)

    dev_examples = processor.get_dev_examples()
    dev_texts = [example.text for example in dev_examples]
    dev_labels = [example.label for example in dev_examples]
    dev_dataset = Data(dev_texts, dev_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.TEST.batch_size)

    test_examples = processor.get_test_examples()
    test_texts = [example.text for example in test_examples]
    test_labels = [example.label for example in test_examples]
    test_dataset = Data(test_texts, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=config.TEST.batch_size)

    return train_dataloader, dev_dataloader, test_dataloader


def regurlarization(list_views, config):
    n_views = config.n_views  # h
    # d_views = config.view_dim  # d
    if len(list_views) < 1 or n_views < 2:
        return torch.tensor(0).to(config.device)
    else:
        penalty = []
        penalty_mask = (torch.ones([n_views, n_views]) - torch.eye(n_views)).to(config.device)
        for views in list_views:
            # views (bs, h, d)
            views_prime = views.transpose(1, 2)  # (bs, d, h)
            penalty_map = views @ views_prime
            # penalty_term = torch.mean(torch.log(penalty_map) * penalty_mask) / 2.
            penalty_term = torch.mean(torch.neg(torch.log(torch.sigmoid(-penalty_map)) * penalty_mask)) / 2.
            penalty.append(penalty_term)
        return torch.tensor(penalty).mean()
