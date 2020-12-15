# coding: UTF-8
# @Author : szbw
# @Email : 20S003015@stu.hit.edu.cn
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForNextSentencePrediction
# from pytorch_pretrained import BertModel, BertTokenizer


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # self.bert = BertForMaskedLM.from_pretrained(config.bert_path)

        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        # self.lstm = nn.LSTM(config.hidden_size, 512, 2,
        #                     bidirectional=True, batch_first=True, dropout=0.1)
        # self.dropout = nn.Dropout(0.1)
        # self.fc_rnn = nn.Linear(512 * 2, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        segment_idx = x[3]
        # _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        encoder, pooled = self.bert(context, token_type_ids=segment_idx, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        # out, _ = self.lstm(encoder)
        # out = self.dropout(out)
        # out = self.fc_rnn(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
