#!/usr/bin/python38

from transformers import BertConfig

BERT_BASE_ZHCN_SIZE = 21128
# num_labels defines the output dimension of BertModel.
bert_config = BertConfig(vocab_size=BERT_BASE_ZHCN_SIZE, num_labels=23)
