## Learning Sentiment Sentence Representation with Multiview Attention Model

This repo contains PyTorch deep learning models for sentiment classification.

## Requirement

```
transformers
yaml
easydict
torchtext
spacy
```



## Usage

- running a model over glove

```
# dataset: sst5 subj mr cr ags ec reuters
# trainer: glove (over glove) bert (over bert)

python run.py --run train --dataset sst5 --model single --trainer glove
```

- running a model over bert

```
python run.py --run train --dataset sst5 --model single --trainer bert
```

## Noting

All training Hyper-Parameters are set in cfgs/config.py and model parameters in cfgs/single.yml.
