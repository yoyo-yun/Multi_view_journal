import io
import os
import torch
import pickle
from functools import partial
from torchtext import data, datasets
from torchtext.data import Example, Dataset
from torchtext.utils import unicode_csv_reader

vectors_cache = os.path.expanduser('~') + r'/.vector_cache'

device = 'cuda'


class MR(Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        for label in ['pos', 'neg']:
            fname = os.path.join(path, 'rt-polarity.{}'.format(label))
            with io.open(fname, 'r', encoding="windows-1252") as f:
                texts = f.readlines()
            for text in texts:
                examples.append(data.Example.fromlist([text, label], fields))

        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path=None, root='.data', **kwargs):
        dataset = cls(path, **kwargs)
        return dataset


class SST2(Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        format = format.lower()
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format]

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == 'csv':
                reader = unicode_csv_reader(f, **csv_reader_params)
            elif format == 'tsv':
                reader = unicode_csv_reader(f, delimiter='\t', **csv_reader_params)
            else:
                reader = f

            if format in ['csv', 'tsv'] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError('When using a dict to specify fields with a {} file,'
                                     'skip_header must be False and'
                                     'the file must have a header.'.format(format))
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(SST2, self).__init__(examples, fields, **kwargs)


class Subj(Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        for label in ['plot', 'quote']:
            fname = os.path.join(path, '{}.tok.gt9.5000'.format(label))
            with io.open(fname, 'r', encoding="windows-1252") as f:
                texts = f.readlines()
            for text in texts:
                examples.append(data.Example.fromlist([text, label], fields))

        super(Subj, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path=None, root='.datasets', **kwargs):
        dataset = cls(path, **kwargs)
        return dataset


class CR(Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        for label in ['pos', 'neg']:
            fname = os.path.join(path, 'custrev.{}'.format(label))
            with io.open(fname, 'r', encoding="windows-1252") as f:
                texts = f.readlines()
            for text in texts:
                examples.append(data.Example.fromlist([text, label], fields))

        super(CR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path=None, root='.datasets', **kwargs):
        dataset = cls(path, **kwargs)
        return dataset


class TREC(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fname, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        fname = os.path.join(path, fname)
        with io.open(fname, 'r', encoding="windows-1252") as f:
            texts = f.readlines()
        for text in texts:
            examples.append(data.Example.fromlist([text[2:], text[0]], fields))
        super(TREC, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path='.datasets/trec', text_field=None, label_field=None,
               train='trec.train.all', validation=None, test='trec.test.all',
               **kwargs):
        train_data = None if train is None else cls(
            path, train, text_field, label_field, **kwargs)
        val_data = None if validation is None else cls(
            path, validation, text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            path, test, text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class AGs(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fname, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        fname = os.path.join(path, fname)
        with io.open(fname, 'r', encoding="windows-1252") as f:
            texts = f.readlines()
        for text in texts:
            examples.append(data.Example.fromlist([text[2:], text[0]], fields))
        super(AGs, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path='.datasets/ags', text_field=None, label_field=None,
               train='agnews.test.all', validation=None, test='agnews.train.all',
               train_subtrees=False, **kwargs):
        train_data = None if train is None else cls(
            path, train, text_field, label_field, **kwargs)
        val_data = None if validation is None else cls(
            path, validation, text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            path, test, text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class EC(Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields, skip_header=True,
                 csv_reader_params={}, **kwargs):
        make_example = Example.fromCSV

        with io.open(os.path.expanduser(path), encoding="utf-8") as f:
            reader = unicode_csv_reader(f, delimiter='\t', **csv_reader_params)
            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(EC, self).__init__(examples, fields, **kwargs)


class SST5Processor(object):
    NAME = 'sst5'
    NUM_CLASSES = 5
    IS_MULTILABEL = False
    train_iterator = None
    dev_iterator = None
    test_iterator = None
    pretrained_embedding = None
    pad_idx = None
    sample = ["I love you more!",
              "You love me.",
              "it virtually defines a comedy that 's strongly mediocre , with funny bits surfacing every once in a while ."]
    sample_idx = None

    def __init__(self, batchsz=32, device=device, include_lengths=True, fix_length=None):
        TEXT = data.Field(batch_first=True, include_lengths=include_lengths, fix_length=fix_length, tokenize='spacy')
        LABEL = data.LabelField()  # multi-classification

        train, val, test = datasets.SST.splits(TEXT, LABEL, fine_grained=True)

        TEXT.build_vocab(train, vectors='glove.840B.300d', vectors_cache=vectors_cache)
        LABEL.build_vocab(train)

        if self.sample is not None:
            self.sample = [TEXT.preprocess(sample) for sample in self.sample]
            self.sample_idx = TEXT.process(self.sample)

        train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
            (train, val, test),
            batch_size=batchsz,
            device=device)
        pretrained_embedding = TEXT.vocab.vectors

        pad_idx = TEXT.vocab.stoi.get('<pad>')
        print("index of <pad> is :" + str(pad_idx))
        self.pad_idx = pad_idx
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator
        self.pretrained_embedding = pretrained_embedding
        self.pad_idx = pad_idx

    def get_train_examples(self):
        return self.train_iterator

    def get_dev_examples(self):
        return self.dev_iterator

    def get_test_examples(self):
        return self.test_iterator

    def get_pred_embed(self):
        return self.pretrained_embedding

    def get_pad_idx(self):
        return self.pad_idx

    def get_sample(self):
        return (self.sample, self.sample_idx)  # ([sample], ([sample_text],[sample_length]))


class TRECProcessor(object):
    NAME = 'trec'
    NUM_CLASSES = 6
    IS_MULTILABEL = False
    train_iterator = None
    dev_iterator = None
    test_iterator = None
    pretrained_embedding = None
    pad_idx = None
    sample = ["What is the only artery that carries blue blood from the heart to the lungs ?",
              "What is the difference between AM radio stations and FM radio stations ?",
              "What color does litmus paper turn when it comes into contact with a strong acid ?"]
    sample_idx = None

    def __init__(self, batchsz=32, device=device, include_lengths=True, fix_length=None):
        TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
        LABEL = data.LabelField()

        train, test = TREC.splits(path=os.path.join('datasets', 'trec'), text_field=TEXT, label_field=LABEL,
                                  train='TREC.train.all', validation=None, test='TREC.test.all')
        val = test
        TEXT.build_vocab(train, vectors='glove.840B.300d', vectors_cache=vectors_cache)
        LABEL.build_vocab(train)

        if self.sample is not None:
            self.sample = [TEXT.preprocess(sample) for sample in self.sample]
            self.sample_idx = TEXT.process(self.sample)

        train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
            (train, val, test),
            batch_size=batchsz,
            device=device)
        pretrained_embedding = TEXT.vocab.vectors

        pad_idx = TEXT.vocab.stoi.get('<pad>')
        print("index of <pad> is :" + str(pad_idx))
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator
        self.pretrained_embedding = pretrained_embedding
        self.pad_idx = pad_idx

    def get_train_examples(self):
        return self.train_iterator

    def get_dev_examples(self):
        return self.dev_iterator

    def get_test_examples(self):
        return self.test_iterator

    def get_pred_embed(self):
        return self.pretrained_embedding

    def get_pad_idx(self):
        return self.pad_idx

    def get_sample(self):
        return (self.sample, self.sample_idx)  # ([sample], ([sample_text],[sample_length]))


class MRProcessor(object):
    NAME = 'mr'
    NUM_CLASSES = 2
    IS_MULTILABEL = False
    train_iterator = None
    dev_iterator = None
    test_iterator = None
    pretrained_embedding = None
    pad_idx = None
    state = pickle.load(open(r'datasets/mr/random_state.pickle3', 'rb'))
    sample = ["I love you more!",
              "You love me.",
              "it virtually defines a comedy that 's strongly mediocre , with funny bits surfacing every once in a while ."]
    sample_idx = None

    def __init__(self, batchsz=32, device=device, include_lengths=True, fix_length=None):
        TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
        LABEL = data.LabelField(dtype=torch.long)

        dataset = MR.splits(path=os.path.join('datasets', 'mr'), text_field=TEXT, label_field=LABEL)
        TEXT.build_vocab(dataset, vectors='glove.840B.300d', vectors_cache=vectors_cache)
        LABEL.build_vocab(dataset)

        if self.sample is not None:
            self.sample = [TEXT.preprocess(sample) for sample in self.sample]
            self.sample_idx = TEXT.process(self.sample)

        train_data, val_data, test_data = dataset.split(split_ratio=[0.8, 0.1, 0.1], random_state=self.state)

        train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size=batchsz,
            device=device)
        pretrained_embedding = TEXT.vocab.vectors

        pad_idx = TEXT.vocab.stoi.get('<pad>')
        print("index of <pad> is :" + str(pad_idx))
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator
        self.pretrained_embedding = pretrained_embedding
        self.pad_idx = pad_idx

    def get_train_examples(self):
        return self.train_iterator

    def get_dev_examples(self):
        return self.dev_iterator

    def get_test_examples(self):
        return self.test_iterator

    def get_pred_embed(self):
        return self.pretrained_embedding

    def get_pad_idx(self):
        return self.pad_idx

    def get_sample(self):
        return (self.sample, self.sample_idx)  # ([sample], ([sample_text],[sample_length]))


class AGsProcessor(object):
    NAME = 'ags'
    NUM_CLASSES = 4
    IS_MULTILABEL = False
    train_iterator = None
    dev_iterator = None
    test_iterator = None
    pretrained_embedding = None
    pad_idx = None
    sample = ["E-mail scam targets police chief Wiltshire Police warns about phishing after its fraud squad chief was targeted.",
              "A Stereo with a Brain You can train Bose's new system to play songs you like. Is it worth the price?"]
    sample_idx = None

    def __init__(self, batchsz=32, device=device, include_lengths=True, fix_length=None):
        TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
        LABEL = data.LabelField()

        train, test = TREC.splits(path=os.path.join('datasets', 'ags'), text_field=TEXT, label_field=LABEL,
                                  train='agnews.train.all', validation=None, test='agnews.test.all')
        val = test
        TEXT.build_vocab(train, vectors='glove.840B.300d', vectors_cache=vectors_cache)
        LABEL.build_vocab(train)

        if self.sample is not None:
            self.sample = [TEXT.preprocess(sample) for sample in self.sample]
            self.sample_idx = TEXT.process(self.sample)

        train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
            (train, val, test),
            batch_size=batchsz,
            device=device)
        pretrained_embedding = TEXT.vocab.vectors

        pad_idx = TEXT.vocab.stoi.get('<pad>')
        print("index of <pad> is :" + str(pad_idx))
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator
        self.pretrained_embedding = pretrained_embedding
        self.pad_idx = pad_idx

    def get_train_examples(self):
        return self.train_iterator

    def get_dev_examples(self):
        return self.dev_iterator

    def get_test_examples(self):
        return self.test_iterator

    def get_pred_embed(self):
        return self.pretrained_embedding

    def get_pad_idx(self):
        return self.pad_idx

    def get_sample(self):
        return (self.sample, self.sample_idx)  # ([sample], ([sample_text],[sample_length]))


class SubjProcessor(object):
    NAME = 'subj'
    NUM_CLASSES = 2
    IS_MULTILABEL = False
    train_iterator = None
    dev_iterator = None
    test_iterator = None
    pretrained_embedding = None
    pad_idx = None
    state = pickle.load(open(r'datasets/subj/random_state.pickle3', 'rb'))
    sample = ["I love you more!",
              "You love me.",
              "it virtually defines a comedy that 's strongly mediocre , with funny bits surfacing every once in a while ."]
    sample_idx = None

    def __init__(self, batchsz=32, device=device, include_lengths=True, fix_length=None):
        TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
        LABEL = data.LabelField(dtype=torch.long)

        dataset = Subj.splits(path=os.path.join('datasets', 'subj'), text_field=TEXT, label_field=LABEL)
        TEXT.build_vocab(dataset, vectors='glove.840B.300d', vectors_cache=vectors_cache)
        LABEL.build_vocab(dataset)

        if self.sample is not None:
            self.sample = [TEXT.preprocess(sample) for sample in self.sample]
            self.sample_idx = TEXT.process(self.sample)

        train_data, val_data, test_data = dataset.split(split_ratio=[0.8, 0.1, 0.1], random_state=self.state)

        train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size=batchsz,
            device=device)
        pretrained_embedding = TEXT.vocab.vectors

        pad_idx = TEXT.vocab.stoi.get('<pad>')
        print("index of <pad> is :" + str(pad_idx))
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator
        self.pretrained_embedding = pretrained_embedding
        self.pad_idx = pad_idx

    def get_train_examples(self):
        return self.train_iterator

    def get_dev_examples(self):
        return self.dev_iterator

    def get_test_examples(self):
        return self.test_iterator

    def get_pred_embed(self):
        return self.pretrained_embedding

    def get_pad_idx(self):
        return self.pad_idx

    def get_sample(self):
        return (self.sample, self.sample_idx)  # ([sample], ([sample_text],[sample_length]))


class CRProcessor(object):
    NAME = 'cr'
    NUM_CLASSES = 2
    IS_MULTILABEL = False
    train_iterator = None
    dev_iterator = None
    test_iterator = None
    pretrained_embedding = None
    pad_idx = None
    state = pickle.load(open(r'datasets/cr/random_state.pickle3', 'rb'))
    sample = ["of course , i was hesitant given the price , but i 've been extremely impressed since receiving it , and bought a second one as a christmas present for my parents ."]
    sample_idx = None

    def __init__(self, batchsz=32, device=device, include_lengths=True, fix_length=None):
        TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
        LABEL = data.LabelField(dtype=torch.long)

        dataset = CR.splits(path=os.path.join('datasets', 'cr'), text_field=TEXT, label_field=LABEL)
        TEXT.build_vocab(dataset, vectors='glove.840B.300d', vectors_cache=vectors_cache)
        LABEL.build_vocab(dataset)

        if self.sample is not None:
            self.sample = [TEXT.preprocess(sample) for sample in self.sample]
            self.sample_idx = TEXT.process(self.sample)

        train_data, val_data, test_data = dataset.split(split_ratio=[0.8, 0.1, 0.1], random_state=self.state)

        train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size=batchsz,
            device=device)
        pretrained_embedding = TEXT.vocab.vectors

        pad_idx = TEXT.vocab.stoi.get('<pad>')
        print("index of <pad> is :" + str(pad_idx))
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator
        self.pretrained_embedding = pretrained_embedding
        self.pad_idx = pad_idx

    def get_train_examples(self):
        return self.train_iterator

    def get_dev_examples(self):
        return self.dev_iterator

    def get_test_examples(self):
        return self.test_iterator

    def get_pred_embed(self):
        return self.pretrained_embedding

    def get_pad_idx(self):
        return self.pad_idx

    def get_sample(self):
        return (self.sample, self.sample_idx)  # ([sample], ([sample_text],[sample_length]))


class ECProcessor(object):
    NAME = 'ec'
    NUM_CLASSES = 11
    IS_MULTILABEL = True
    train_iterator = None
    dev_iterator = None
    test_iterator = None
    pretrained_embedding = None
    pad_idx = None
    # sample = ["For every one concept I try to cram into my brain, I feel like I'm pushing out three."]
    # sample = ["Whatever you decide to do make sure it makes you #happy."]
    # sample = ["Oh dear an evening of absolute hilarity I don't think I have laughed so much in a long time!"]
    # sample = ["At the end of the day I know that my kids will never worry about me leaving them"]
    sample = ["Tutoring gives me such an exhilarating feeling. I love helping people"]
    sample_idx = None

    def __init__(self, batchsz=32, device=device, include_lengths=True, fix_length=None):
        label_type = torch.float
        ID = None
        TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
        ANGER = data.LabelField(dtype=label_type)
        ANTICIPATION = data.LabelField(dtype=label_type)
        DISGUST = data.LabelField(dtype=label_type)
        FEAR = data.LabelField(dtype=label_type)
        JOY = data.LabelField(dtype=label_type)
        LOVE = data.LabelField(dtype=label_type)
        OPTIMISM = data.LabelField(dtype=label_type)
        PESSIMISM = data.LabelField(dtype=label_type)
        SADNESS = data.LabelField(dtype=label_type)
        SURPRISE = data.LabelField(dtype=label_type)
        TRUST = data.LabelField(dtype=label_type)
        fields = [('id', None),
                  ('text', TEXT),
                  ('anger', ANGER),
                  ('anticipation', ANTICIPATION),
                  ('disgust', DISGUST),
                  ('fear', FEAR),
                  ('joy', JOY),
                  ('love', LOVE),
                  ('optimism', OPTIMISM),
                  ('pessimism', PESSIMISM),
                  ('sadness', SADNESS),
                  ('surprise', SURPRISE),
                  ('trust', TRUST)
                  ]

        train, dev, test = EC.splits(path=os.path.join('datasets', 'ec'), train='2018-E-c-En-train.txt', validation='2018-E-c-En-dev.txt',
                                     test='2018-E-c-En-test-gold.txt', fields=fields)
        TEXT.build_vocab(train, vectors='glove.840B.300d', vectors_cache=vectors_cache)
        for i in range(2, len(fields)):
            fields[i][1].build_vocab(train)

        if self.sample is not None:
            self.sample = [TEXT.preprocess(sample) for sample in self.sample]
            self.sample_idx = TEXT.process(self.sample)

        train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
            (train, dev, test),
            batch_size=batchsz,
            device=device)
        pretrained_embedding = TEXT.vocab.vectors

        pad_idx = TEXT.vocab.stoi.get('<pad>')
        print("index of <pad> is :" + str(pad_idx))
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator
        self.pretrained_embedding = pretrained_embedding
        self.pad_idx = pad_idx

    def get_train_examples(self):
        return self.train_iterator

    def get_dev_examples(self):
        return self.dev_iterator

    def get_test_examples(self):
        return self.test_iterator

    def get_pred_embed(self):
        return self.pretrained_embedding

    def get_pad_idx(self):
        return self.pad_idx

    def get_sample(self):
        return (self.sample, self.sample_idx)  # ([sample], ([sample_text],[sample_length]))

