import os
import io
import csv
import sys
import random
import pickle
from torch.utils.data import Dataset

vectors_cache = os.path.expanduser('~') + r'/.vector_cache'


def check_split_ratio(split_ratio):
    """Check that the split ratio argument is not malformed"""
    valid_ratio = 0.
    if isinstance(split_ratio, float):
        # Only the train set relative ratio is provided
        # Assert in bounds, validation size is zero
        assert 0. < split_ratio < 1., (
            "Split ratio {} not between 0 and 1".format(split_ratio))

        test_ratio = 1. - split_ratio
        return (split_ratio, test_ratio, valid_ratio)
    elif isinstance(split_ratio, list):
        # A list of relative ratios is provided
        length = len(split_ratio)
        assert length == 2 or length == 3, (
            "Length of split ratio list should be 2 or 3, got {}".format(split_ratio))

        # Normalize if necessary
        ratio_sum = sum(split_ratio)
        if not ratio_sum == 1.:
            split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]

        if length == 2:
            return tuple(split_ratio + [valid_ratio])
        return tuple(split_ratio)
    else:
        raise ValueError('Split ratio must be float or a list, got {}'
                         .format(type(split_ratio)))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class Data(Dataset):
    def __init__(self, *data):
        assert all(len(data[0]) == len(d) for d in data)
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)


class SST2Processor(object):
    NAME = 'SST2'
    NUM_CLASSES = 2
    IS_MULTILABEL = False
    MAX_SEQ_LENGTH = 128

    def get_train_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'SST2', 'sentiment-train')), 'train')

    def get_dev_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'SST2', 'sentiment-dev')), 'dev')

    def get_test_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'SST2', 'sentiment-test')), 'test')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line[0]
            label = int(line[1])
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class SST5Processor(object):
    NAME = 'SST5'
    NUM_CLASSES = 5
    IS_MULTILABEL = False
    MAX_SEQ_LENGTH = 80

    # 80 4 25.1

    def get_train_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'sst5', 'train.txt')), 'train')

    def get_dev_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'sst5', 'dev.txt')), 'dev')

    def get_test_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'sst5', 'test.txt')), 'test')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line[0]
            label = int(line[1])
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_tsv(cls, input_file):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        with open(input_file, "r") as f:
            # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            examples = [cls.fromtree(line) for line in f]
            return examples

    def fromtree(self, data):
        try:
            from nltk.tree import Tree
        except ImportError:
            print("Please install NLTK. "
                  "See the docs at http://nltk.org for more information.")
            raise
        # print(data)
        tree = Tree.fromstring(data)
        text = ' '.join(tree.leaves())
        label = tree.label()
        return text, label


class MRProcessor(object):
    NAME = 'MR'
    NUM_CLASSES = 2
    IS_MULTILABEL = False
    random_state = pickle.load(open(r'datasets/mr/random_state.pickle3', 'rb'))
    MAX_SEQ_LENGTH = 78

    # 78 3 27.4

    def get_train_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_datasets(os.path.join(data_dir, 'mr')), 'train')

    def get_dev_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_datasets(os.path.join(data_dir, 'mr')), 'dev')

    def get_test_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_datasets(os.path.join(data_dir, 'mr')), 'test')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[set_type]):
            guid = "%s-%s" % (set_type, i)
            text = line[0]
            label = int(line[1])
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_datasets(self, file_dir):
        examples = []
        for label in ['pos', 'neg']:
            fname = os.path.join(file_dir, 'rt-polarity.{}'.format(label))
            with io.open(fname, 'r', encoding="windows-1252") as f:
                texts = f.readlines()
                examples += [(text, 1 if label == 'pos' else 0) for text in texts]
        train_examples, dev_examples, test_examples = self._split(file_dir,
                                                                  examples,
                                                                  split_ratio=[0.8, 0.1, 0.1],
                                                                  random_state=self.random_state)
        return {
            "train": train_examples,
            'dev': dev_examples,
            'test': test_examples
        }

    def _split(self, file_dir, examples, split_ratio, random_state=None):
        if random_state is None:
            self.random_state = random.getstate()
        random.setstate(self.random_state)
        # assert random_state == self.random_state, "different random state may lead an unbelievable test prediction"
        with open(os.path.join(file_dir, 'random_state.pickle3'), 'wb') as f:
            pickle.dump(self.random_state, f)
        train_ratio, test_ratio, val_ratio = check_split_ratio(split_ratio)
        N = len(examples)
        randperm = random.sample(range(N), N)
        train_len = int(round(train_ratio * N))

        # Due to possible rounding problems
        if not val_ratio:
            test_len = N - train_len
        else:
            test_len = int(round(test_ratio * N))

        indices = (randperm[:train_len],  # Train
                   randperm[train_len:train_len + test_len],  # Test
                   randperm[train_len + test_len:])  # Validation

        # There's a possibly empty list for the validation set
        data = tuple([examples[i] for i in index] for index in indices)
        return data


class TRECProcessor(object):
    NAME = 'TREC'
    NUM_CLASSES = 6
    IS_MULTILABEL = False
    MAX_SEQ_LENGTH = 41

    # 41 5 13.0

    def get_train_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'trec', 'TREC.train.all')), 'train')

    def get_dev_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'trec', 'TREC.test.all')), 'test')

    def get_test_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'trec', 'TREC.test.all')), 'test')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line[2:]
            label = int(line[0])
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_tsv(cls, input_file):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        with io.open(input_file, 'r', encoding="windows-1252") as f:
            lines = f.readlines()
            return [line for line in lines]


class AGsProcessor(object):
    NAME = 'AGs'
    NUM_CLASSES = 4
    IS_MULTILABEL = False
    MAX_SEQ_LENGTH = 128

    # 348 15 54.6

    def get_train_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'ags', 'agnews.train.all')), 'train')

    def get_dev_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'ags', 'agnews.test.all')), 'test')

    def get_test_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'ags', 'agnews.test.all')), 'test')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line[2:]
            label = int(line[0]) - 1
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_tsv(cls, input_file):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        with io.open(input_file, 'r', encoding="windows-1252") as f:
            lines = f.readlines()
            return [line for line in lines]


class SUBJProcessor(object):
    NAME = 'Subj'
    NUM_CLASSES = 2
    IS_MULTILABEL = False
    random_state = pickle.load(open(r'datasets/subj/random_state.pickle3', 'rb'))
    # random_state = random.getstate()
    MAX_SEQ_LENGTH = 78

    # 78 3 27.4

    def get_train_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_datasets(os.path.join(data_dir, 'subj')), 'train')

    def get_dev_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_datasets(os.path.join(data_dir, 'subj')), 'dev')

    def get_test_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_datasets(os.path.join(data_dir, 'subj')), 'test')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[set_type]):
            guid = "%s-%s" % (set_type, i)
            text = line[0]
            label = int(line[1])
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_datasets(self, file_dir):
        examples = []
        for label in ['plot', 'quote']:
            fname = os.path.join(file_dir, '{}.tok.gt9.5000'.format(label))
            with io.open(fname, 'r', encoding="windows-1252") as f:
                texts = f.readlines()
                examples += [(text, 1 if label == 'plot' else 0) for text in texts]
        train_examples, dev_examples, test_examples = self._split(file_dir,
                                                                  examples,
                                                                  split_ratio=[0.8, 0.1, 0.1],
                                                                  random_state=self.random_state)
        return {
            "train": train_examples,
            'dev': dev_examples,
            'test': test_examples
        }

    def _split(self, file_dir, examples, split_ratio, random_state=None):
        if random_state is None:
            self.random_state = random.getstate()
        random.setstate(self.random_state)
        # assert random_state == self.random_state, "different random state may lead an unbelievable test prediction"
        with open(os.path.join(file_dir, 'random_state.pickle3'), 'wb') as f:
            pickle.dump(self.random_state, f)
        train_ratio, test_ratio, val_ratio = check_split_ratio(split_ratio)
        N = len(examples)
        randperm = random.sample(range(N), N)
        train_len = int(round(train_ratio * N))

        # Due to possible rounding problems
        if not val_ratio:
            test_len = N - train_len
        else:
            test_len = int(round(test_ratio * N))

        indices = (randperm[:train_len],  # Train
                   randperm[train_len:train_len + test_len],  # Test
                   randperm[train_len + test_len:])  # Validation

        # There's a possibly empty list for the validation set
        data = tuple([examples[i] for i in index] for index in indices)
        return data


class CRProcessor(object):
    NAME = 'CR'
    NUM_CLASSES = 2
    IS_MULTILABEL = False
    random_state = pickle.load(open(r'datasets/cr/random_state.pickle3', 'rb'))
    MAX_SEQ_LENGTH = 78

    # 78 3 27.4

    def get_train_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_datasets(os.path.join(data_dir, 'cr')), 'train')

    def get_dev_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_datasets(os.path.join(data_dir, 'cr')), 'dev')

    def get_test_examples(self, data_dir='datasets'):
        return self._create_examples(
            self._read_datasets(os.path.join(data_dir, 'cr')), 'test')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[set_type]):
            guid = "%s-%s" % (set_type, i)
            text = line[0]
            label = int(line[1])
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_datasets(self, file_dir):
        examples = []
        for label in ['neg', 'pos']:
            fname = os.path.join(file_dir, 'custrev.{}'.format(label))
            with io.open(fname, 'r', encoding="windows-1252") as f:
                texts = f.readlines()
                examples += [(text, 1 if label == 'pos' else 0) for text in texts]
        train_examples, dev_examples, test_examples = self._split(file_dir,
                                                                  examples,
                                                                  split_ratio=[0.8, 0.1, 0.1],
                                                                  random_state=self.random_state)
        return {
            "train": train_examples,
            'dev': dev_examples,
            'test': test_examples
        }

    def _split(self, file_dir, examples, split_ratio, random_state=None):
        if random_state is None:
            self.random_state = random.getstate()
        random.setstate(self.random_state)
        # assert random_state == self.random_state, "different random state may lead an unbelievable test prediction"
        with open(os.path.join(file_dir, 'random_state.pickle3'), 'wb') as f:
            pickle.dump(self.random_state, f)
        train_ratio, test_ratio, val_ratio = check_split_ratio(split_ratio)
        N = len(examples)
        randperm = random.sample(range(N), N)
        train_len = int(round(train_ratio * N))

        # Due to possible rounding problems
        if not val_ratio:
            test_len = N - train_len
        else:
            test_len = int(round(test_ratio * N))

        indices = (randperm[:train_len],  # Train
                   randperm[train_len:train_len + test_len],  # Test
                   randperm[train_len + test_len:])  # Validation

        # There's a possibly empty list for the validation set
        data = tuple([examples[i] for i in index] for index in indices)
        return data
