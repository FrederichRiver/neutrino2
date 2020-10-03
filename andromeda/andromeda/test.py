#!/usr/bin/python38
from torch.nn import Softmax
from andromeda.model import tokenizer, bert_model, Features

label_list = [
    "X", 'B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
    'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
    'O', 'S-NAME', 'S-ORG', 'S-RACE', "[START]", "[END]"]

raw = "在9月12日首届中国金融四十人“曲江论坛”上，全国政协委员肖钢表示，我国成为全球跨境投资的大国，已经成为净资本输出国。这是中国资本寻求全球机会的需要，更是资本吸收国主动谋求发展的需要。对外投资是企业全球配置资源、产业布局、靠近生产，实现持续稳健发展的有效的途径，也是构建国内经济大循环为主体，国内国际双循环相互促进新发展格局的有力保障。对外投资规模不断扩大，取得明显成就的同时，海外投资权益保护的重要性、紧迫性也日益凸显出来。"

# token = {'input_ids':[word vector list], 'token_type_ids': [list], 'attention_mask': [list] }
token = tokenizer(raw)
# print(token)
feature_list = []
token_list = []
token_list.append(token['input_ids'])
for token_ids in token_list:
    feature_list.append(Features(token_ids, len(token_ids)))
# print(feature_list)

# token = (input_ids, attention_mask, labels, input_lens)

output = bert_model(feature_list[0].input_ids)

x = output[0].view(512, -1)

softmax = Softmax(dim=1)
result = softmax(x)
print(result)
