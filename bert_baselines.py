# coding: UTF-8
# @Author : szbw
# @Email : 20S003015@stu.hit.edu.cn
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert import BertModel, BertConfig, file_utils
import copy
from transformers import AutoModel, RobertaModel, AutoConfig, BertModel, AlbertModel,BertConfig


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.hidden = config.hidden_size
        # self.bert = BertForMaskedLM.from_pretrained(config.bert_path)
        self.model_name = config.bert_path
        if 'albert' in config.bert_path:
            self.bert = AlbertModel.from_pretrained(config.bert_path)
        elif 'robert' in config.bert_path or 'nezha' in config.bert_path:
            self.bert = BertModel.from_pretrained(config.bert_path)
        else:
            self.bert = AutoModel.from_pretrained(config.bert_path)
        self.pooling_mode_max_tokens = True
        self.pooling_mode_mean_tokens = True
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.hidden*3, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        context = x[0]  # 输入的句子
        bsz = context.shape[0]
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        segment_idx = x[3]
        output_vectors = []
        # _, pooled = self.bert(context, attention_mask=mask, token_type_ids=segment_idx, output_all_encoded_layers=False)
        if 'electra' in self.model_name:
            outputs = self.bert(context, token_type_ids=segment_idx, attention_mask=mask)
            encoder = outputs[0]
        else:
            encoder, pooled = self.bert(context, token_type_ids=segment_idx, attention_mask=mask)

        if self.pooling_mode_mean_tokens:
            input_mask_expanded = mask.unsqueeze(-1).expand(encoder.size()).float()
            sum_embeddings = torch.sum(encoder * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.insert(0, sum_embeddings / sum_mask)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = mask.unsqueeze(-1).expand(encoder.size()).float()
            encoder[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(encoder, 1)[0]
            output_vectors.insert(1, max_over_time)

        output_vector = torch.cat(output_vectors, 1)
        output_vector = self.dropout(output_vector)
        out = self.fc(output_vector)
        return out