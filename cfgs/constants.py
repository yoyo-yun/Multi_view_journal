from models import encoder
from datasets import dataset_from_Bert
from datasets import dataset_from_torchtxt
from models import base_net, encoder2decoder, encoder2decoder_v2, baselines

DATASET_TORCHTXT = {
    # 'sst2': dataset_from_torchtxt.SST2Processor,
    'sst5': dataset_from_torchtxt.SST5Processor,
    'mr': dataset_from_torchtxt.MRProcessor,
    'trec': dataset_from_torchtxt.TRECProcessor,
    'ags': dataset_from_torchtxt.AGsProcessor,
    'subj': dataset_from_torchtxt.SubjProcessor,
    'cr': dataset_from_torchtxt.CRProcessor,
    'ec': dataset_from_torchtxt.ECProcessor,
}

DATASET_BERT = {
    'sst2': dataset_from_Bert.SST2Processor,
    'sst5': dataset_from_Bert.SST5Processor,
    'mr': dataset_from_Bert.MRProcessor,
    'trec': dataset_from_Bert.TRECProcessor,
    'ags': dataset_from_Bert.AGsProcessor,
    'subj': dataset_from_Bert.SUBJProcessor,
    'cr': dataset_from_Bert.CRProcessor,
}

DATASET_NAME = {
    'sst2': 'sst2',
    'sst5': 'sst5',
    'mr': 'mr',
    'TRED': 'TRED',
    'ags': 'ags',
    'subj': 'subj',
    'cr': 'cr',
    'reuters': 'reuters',
    'ec': 'ec',
}

MODEL = {
    "single": base_net.Net,
    "e2d": encoder2decoder_v2.Net,
    "stack": base_net.Net,
    "lstm": baselines.LSTM,
}

MUL_MODEL={
    "single": base_net.MulNet,
    "lstm": baselines.LSTM
}

ENCODER = {
    "glove_lstm": encoder.GloveLSTMEncoder,
    "bert": encoder.BERTEncoder,
    'refine': encoder.RefineBERT,
    "bert_lstm": encoder.BERTLSTMEncoder,
    "glove": encoder.GloveEncoder
}
