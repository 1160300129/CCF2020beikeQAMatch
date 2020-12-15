import pandas as pd
import matplotlib.pyplot as plt
pad = 230

def statistic_seqlen(path, col, encode):
    with open(path, 'r', encoding=encode) as f:
        b = f.read()
        lines = b.strip().split('\n')
        min_len = 1000000   # 最小句子长度
        max_len = -1000000  # 最大句子长度
        d = {}  # 统计各个长度的句子有多少个
        seq_len = []
        count = 0  # 统计超过阈值pad大小的长度占比为多少，假定pad=32
        for line in lines:
            sentence = line.split('\t')[col]
            length = len(sentence)
            if length <= pad:
                count+=1
            if length == 0:
                print(line.split('\t')[0])
            if length > max_len:
                max_len = length
            if length < min_len:
                min_len = length
            if 0 < length <= 4:
                if '0-5' in d.keys():
                    d['0-5'] += 1
                else:
                    d['0-5'] = 1
            elif 5 < length <= 10:
                if '5-10' in d.keys():
                    d['5-10'] += 1
                else:
                    d['5-10'] = 1
            elif 10 < length <= 15:
                if '10-15' in d.keys():
                    d['10-15'] += 1
                else:
                    d['10-15'] = 1
            else:
                if '>15' in d.keys():
                    d['>15'] += 1
                else:
                    d['>15'] = 1
            seq_len.append(length)
        print("    句子长度超过"+str(pad)+"的占"+str(count / len(seq_len)))
        print('    最小句子长度:' + str(min_len))
        print('    最大句子长度:' + str(max_len))
        print('    ' + str(sum(seq_len)/len(seq_len)))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        x = ['0-5', '5-10', '10-15', '>15']
        y = []
        for key in x:
            # x.append(key)
            y.append(d[key])
        plt.bar(x, y)
        if 'train.query' in path:
            plt.title('训练query分布')
        elif 'train.reply' in path:
            plt.title('训练reply分布')
        elif 'test.query' in path:
            plt.title('测试query分布')
        elif 'test.reply' in path:
            plt.title('测试reply分布')
        plt.show()


def statistic_cls(path):
    with open(path, 'r', encoding='utf8') as f:
        b = f.read()
        lines = b.strip().split('\n')
        d = dict()  # 用来存储id为i的n个回复的1和0
        c_0 = dict()
        c_1 = dict()
        num_1 = 0
        num_0 = 0
        for line in lines:
            idx = line.split('\t')[0]
            cls = line.split('\t')[3]
            length = len(line.split('\t')[2])
            if cls == '1':
                num_1 += 1
                if 0 < length <= 5:
                    if '0-5' in c_0.keys():
                        c_0['0-5'] += 1
                    else:
                        c_0['0-5'] = 1
                elif 5 < length <= 10:
                    if '5-10' in c_0.keys():
                        c_0['5-10'] += 1
                    else:
                        c_0['5-10'] = 1
                elif 10 < length <= 15:
                    if '10-15' in c_0.keys():
                        c_0['10-15'] += 1
                    else:
                        c_0['10-15'] = 1
                else:
                    if '>15' in c_0.keys():
                        c_0['>15'] += 1
                    else:
                        c_0['>15'] = 1
            else:
                num_0 += 1
                if 0 < length <= 5:
                    if '0-5' in c_1.keys():
                        c_1['0-5'] += 1
                    else:
                        c_1['0-5'] = 1
                elif 5 < length <= 10:
                    if '5-10' in c_1.keys():
                        c_1['5-10'] += 1
                    else:
                        c_1['5-10'] = 1
                elif 10 < length <= 15:
                    if '10-15' in c_1.keys():
                        c_1['10-15'] += 1
                    else:
                        c_1['10-15'] = 1
                else:
                    if '>15' in c_1.keys():
                        c_1['>15'] += 1
                    else:
                        c_1['>15'] = 1
            if idx in d.keys():
                d[idx].append(cls)
            else:
                d[idx] = [cls]
        reply_1 = 0  # 回复中有多个1
        reply_0 = 0  # 回复中有全是0
        reply_normal = 0
        for key in d.keys():
            if d[key].count('1') > 1:
                reply_1 += 1
            if d[key].count('0') == len(d[key]):
                reply_0 += 1
            if d[key].count('1') == 1:
                reply_normal += 1
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        x = ['0-5', '5-10', '10-15', '>15']
        y_0 = []
        y_1 = []
        for key in x:
            # x.append(key)
            y_0.append(c_0[key])
            y_1.append(c_1[key])
        plt.title('标签1的回复长度分布')
        plt.bar(x, y_0)
        plt.show()
        plt.title('标签0的回复长度分布')
        plt.bar(x, y_1)
        plt.show()
        print('训练集的所有对话中')
        print('回复全为0的对话有'+str(reply_0)+'组')
        print('回复有多个1的对话'+str(reply_1)+'组')
        print('回复只有一个1的对话'+str(reply_normal)+'组')
        print('标签为0的回复'+str(num_0)+'个')
        print('标签为1的回复'+str(num_1)+'个')


def main():
    print('训练集query')
    statistic_seqlen('train/train.query.tsv', 1, 'utf8')
    print('训练集reply')
    statistic_seqlen('train/final_train.reply.tsv', 2, 'utf8')
    print('测试集query')
    statistic_seqlen('test/test.query.tsv', 1, 'gbk')
    print('测试集reply')
    statistic_seqlen('test/test.reply.tsv', 2, 'gbk')
    statistic_cls('train/final_train.reply.tsv')

    from pytorch_pretrained_bert import BertTokenizer
    # train_left = pd.read_csv('./train/train.query.tsv', sep='\t', header=None)
    # train_left.columns = ['id', 'q1']
    # question = train_left['q1'].tolist()
    #
    # d = dict()
    # import jieba
    # c = 0
    # x = 0
    # for i in range(len(question)):
    #     k = 0
    #     print(x)
    #     q1 = ' '.join(jieba.cut(question[i])).split(' ')
    #     c += len(q1)
    #     for j in range(len(q1)):
    #         if q1[j] not in d.keys():
    #             d[q1[j]] = {i:[1, [k]]}
    #         else:
    #             if i in d[q1[j]]:
    #                 d[q1[j]][i][1].append(k)
    #                 d[q1[j]][i][0] = len(d[q1[j]][i][1])
    #             else:
    #                 d[q1[j]][i] = [1, [k]]
    #         k+=1
    #     x+=1
    # tfidf = dict()
    # import math
    # for key in d.keys():
    #
    #     idf = math.log(6000 / (1 + len(d[key].keys())))
    #     tf = 0
    #     for k1 in d[key].keys():
    #         tf += d[key][k1][0]
    #     tf = tf / c
    #     tfidf[key] = tf * idf
    # res = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)
    # print(111)

    # train_df = pd.read_csv('train/train.query.tsv', sep='\t',header=None, nrows=100)
    # train_df.head()


if __name__ == '__main__':
    main()