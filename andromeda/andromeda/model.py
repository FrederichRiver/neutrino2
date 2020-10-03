#!/usr/bin/python38
import json
import copy
import torch
from transformers import BertForTokenClassification
from transformers.tokenization_bert import BertTokenizer
from .model_config import bert_config


class Features(object):
    """A single set of features of data."""
    Max_seq_len = 512
    Special_tokens_count = 2
    PAD_TOKEN = 0

    def __init__(self, input_ids, input_len):
        self.input_ids = input_ids
        # self.input_mask = input_mask
        # self.segment_ids = segment_ids
        # self.label_ids = label_ids
        self.input_len = input_len
        padding_length = self.Max_seq_len - self.input_len
        self.input_ids += [self.PAD_TOKEN] * padding_length
        self.input_ids = torch.tensor([self.input_ids])

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', config=bert_config)
bert_model = BertForTokenClassification.from_pretrained('bert-base-chinese', config=bert_config)
