"""
BC4CHEMD, BC5CDR-chemを使って薬剤名のNERにfine tuning
$ export COMET_API_KEY=xxxxxx
$ export COMET_PROJECT_NAME=hf_tr_fine_tune_BC4CHEMD

epoch 1, --sampleで回せばとりあえず動くか確認できる
$ python fine_tune_bc4chemd.py [model] [dataset] [epoch] [--sample]
$ python fine_tune_ner.py "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" bc4chemd 1 --sample
$ python fine_tune_ner.py "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" bc5cdr-chem 1 --sample

オリジナルデータ
https://biocreative.bioinformatics.udel.edu/tasks/biocreative-iv/chemdner/
"""
import argparse
from datetime import datetime
from pathlib import Path
import re

import comet_ml
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

DATA_DIR = Path('data/')


def read(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


class BC4CHEMDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model', type=str, default='distilbert-base-cased', choices=[
        'distilbert-base-cased',
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    ])
    parser.add_argument('dataset', type=str, default='bc5cdr-chem', choices=[
        'bc4chemd',
        'bc5cdr-chem'
    ])
    parser.add_argument('epoch', type=int)
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset
    if dataset == 'bc4chemd':
        data_dir = DATA_DIR / 'BC4CHEMD'
    elif dataset == 'bc5cdr-chem':
        data_dir = DATA_DIR / 'BC5CDR-chem'
    else:
        raise ValueError(f'{dataset=} is not available.')
    epoch = args.epoch
    sample = args.sample

    train_texts, train_tags = read(data_dir / 'train.tsv')
    val_texts, val_tags = read(data_dir / 'devel.tsv')
    test_texts, test_tags = read(data_dir / 'train.tsv')
    train_texts = train_texts + val_texts
    train_tags = train_tags + val_tags

    # 長すぎる文章があると失敗するので除外
    # https://github.com/huggingface/transformers/issues/5611
    train_texts = [t for t in train_texts if len(t) < 200]
    train_tags = [t for t in train_tags if len(t) < 200]
    test_texts = [t for t in test_texts if len(t) < 200]
    test_tags = [t for t in test_tags if len(t) < 200]

    # 少量データでチェック
    if sample:
        train_texts = train_texts[:80]
        train_tags = train_tags[:80]
        test_texts = test_texts[:20]
        test_tags = test_tags[:20]

    unique_tags = set(tag for doc in train_tags + test_tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(
        train_texts, is_split_into_words=True, return_offsets_mapping=True,
        padding=True, truncation=True, max_length=1024
    )
    test_encodings = tokenizer(
        test_texts, is_split_into_words=True, return_offsets_mapping=True,
        padding=True, truncation=True, max_length=1024
    )

    train_labels = encode_tags(train_tags, train_encodings)
    test_labels = encode_tags(test_tags, test_encodings)

    train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
    test_encodings.pop("offset_mapping")
    train_dataset = BC4CHEMDDataset(train_encodings, train_labels)
    test_dataset = BC4CHEMDDataset(test_encodings, test_labels)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(unique_tags), id2label=id2tag)
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=epoch,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
    )

    trainer.train()

    trainer.evaluate()

    save_dir = f'models/{dataset}/' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
