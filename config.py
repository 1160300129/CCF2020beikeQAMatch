# @Author : szbw
# @Email : 20S003015@stu.hit.edu.cn
import torch
from transformers import AutoTokenizer, BertTokenizer, AlbertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, ):
        self.model_name = 'nazha-fgm'                                         # 模型名称,微调后的模型保存路径有这个名称
        # self.train_path = 'train/train_data'                                # 训练集
        # self.dev_path = 'test/valid_data'                                    # 验证集
        # self.test_path = 'test/test_data'                                  # 测试集
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        # self.device = 'cpu'
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练 实际训练没用到
        self.num_classes = 2                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 2                                          # mini-batch大小
        self.pad_size = 64                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5                                       # 学习率
        # self.bert_path = './bert_pretrain'
        # self.model_path = './robert'
        self.bert_path = './nezha-cn-base'                            # 本地预训练模型的路径名称
        # # self.bert_path = './electra'
        # self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        if 'albert' in self.bert_path or 'nezha' in self.bert_path or 'roberta':
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768  # bert encoder 输出维度 使用large模型时改成1024
        self.test_batch = 512   # 测试时的batch_size

        # sentence_bert 专用参数
        self.bert_model_path = './distiluse-base-multilingual-cased-v2/0_DistilBERT'
        self.pooling_path = './distiluse-base-multilingual-cased-v2/1_Pooling'
        self.dense_path = './distiluse-base-multilingual-cased-v2/2_Dense'
        self.concatenation_sent_rep = True
        self.concatenation_sent_difference = True
        self.concatenation_sent_multiplication = False
