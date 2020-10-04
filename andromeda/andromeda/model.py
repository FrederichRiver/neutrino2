#!/usr/bin/python38
from transformers import BertForTokenClassification
from transformers.configuration_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from andromeda.crf import CRF


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
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,)+outputs
        return outputs  # (loss), scores
