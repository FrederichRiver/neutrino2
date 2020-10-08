#!/usr/bin/python38
from transformers import BertModel, BertForTokenClassification
from transformers.configuration_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from andromeda.crf import CRF
from torch import nn


class CNerTokenizer(BertTokenizer):
    def tokenize(self, text: list) -> list:
        _tokens = []
        for c in text:
            c = c.lower()
            _tokens.append(c)
        return _tokens


class BertCRFModel(BertForTokenClassification):
    def __init__(self, config: BertConfig) -> None:
        super(BertCRFModel, self).__init__(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # output format: (logits,)
        outputs = (logits,)
        # labels setting
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,)+outputs
        return outputs  # (loss), scores


class BertCRFModel2(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super(BertCRFModel2, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-chinese', config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # output format: (logits,)
        outputs = (logits,)
        # labels setting
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,)+outputs
        return outputs  # (loss), scores
