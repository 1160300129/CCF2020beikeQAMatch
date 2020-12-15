# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_nezha import BertModel, BertConfig
import copy
# from pytorch_pretrained_bert import BertModel


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.hidden = config.hidden_size
        # self.bert = BertForMaskedLM.from_pretrained(config.bert_path)
        self.model_name = config.bert_path
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.pooling_mode_max_tokens = True
        self.pooling_mode_mean_tokens = True
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.hidden*3, 2)
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 256, (k, 1024)) for k in [2, 3, 4]])
        # self.dropout = nn.Dropout(0.5)
        #
        # self.fc_cnn = nn.Linear(256 * 3, 2)
        # self.pooling_mode_max_tokens = True
        # self.pooling_mode_mean_tokens = True
        # self.lstm = nn.LSTM(config.hidden_size, 512, 2,
        #                     bidirectional=True, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.5)
        # self.fc_rnn = nn.Linear(512 * 2, config.num_classes)

    # def conv_and_pool(self, x, conv):
    #     x = F.relu(conv(x)).squeeze(3)
    #     x = F.max_pool1d(x, x.size(2)).squeeze(2)
    #     return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        bsz = context.shape[0]
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        segment_idx = x[3]
        output_vectors = []
        # _, pooled = self.bert(context, attention_mask=mask, token_type_ids=segment_idx, output_all_encoded_layers=False)
        encoder, pooled = self.bert(context, token_type_ids=segment_idx, attention_mask=mask)
        # seq_len = x[1]
        # out = encoder.unsqueeze(1)
        # out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # out = self.dropout(out)
        # out = self.fc_cnn(out)
        # return out
        # t = []
        # for i in range(len(seq_len)):
        #     t.append(encoder[i, seq_len[i]-1,:])
        # output_vectors.append(torch.cat(t, 0).view(bsz, self.hidden))
        output_vectors.append(pooled)
        # output_vectors.append(pooled)
        if self.pooling_mode_mean_tokens:
            input_mask_expanded = mask.unsqueeze(-1).expand(encoder.size()).float()
            sum_embeddings = torch.sum(encoder * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.insert(0, sum_embeddings / sum_mask)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = mask.unsqueeze(-1).expand(encoder.size()).float()
            encoder[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(encoder, 1)
            max_over_time = max_over_time[0]
            output_vectors.insert(1, max_over_time)

        output_vector = torch.cat(output_vectors, 1)
        output_vector = self.dropout(output_vector)
        out = self.fc(output_vector)
        return out
        # prob = F.softmax(out, dim=-1)
        # out, _ = self.lstm(encoder)
        # out = self.dropout(out)
        # out = self.fc_rnn(out[:, -1, :])  # 句子最后时刻的 hidden state