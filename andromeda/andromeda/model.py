#!/usr/bin/python38
import json
import copy
import torch
from transformers import BertForTokenClassification, BertConfig
from transformers.tokenization_bert import BertTokenizer


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


BERT_BASE_ZHCN_SIZE = 21128
bert_config = BertConfig(vocab_size=BERT_BASE_ZHCN_SIZE, num_labels=23)
label_list = [
    "X", 'B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
    'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
    'O', 'S-NAME', 'S-ORG', 'S-RACE', "[START]", "[END]"]

raw = "在9月12日首届中国金融四十人“曲江论坛”上，全国政协委员肖钢表示，我国成为全球跨境投资的大国，已经成为净资本输出国。这是中国资本寻求全球机会的需要，更是资本吸收国主动谋求发展的需要。对外投资是企业全球配置资源、产业布局、靠近生产，实现持续稳健发展的有效的途径，也是构建国内经济大循环为主体，国内国际双循环相互促进新发展格局的有力保障。对外投资规模不断扩大，取得明显成就的同时，海外投资权益保护的重要性、紧迫性也日益凸显出来。"

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',config=bert_config)
# token = {'input_ids':[word vector list], 'token_type_ids': [list], 'attention_mask': [list] }
token = tokenizer(raw)
# print(token)
feature_list = []
token_list = []
token_list.append(token['input_ids'])
for token_ids in token_list:
    feature_list.append(Features(token_ids, len(token_ids)))
# print(feature_list)
model = BertForTokenClassification.from_pretrained('bert-base-chinese')

# input_ids, attention_mask, labels, input_lens

output = model(feature_list[0].input_ids)

print(output)
