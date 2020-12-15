import pandas as pd
import random
import numpy as np
from transformers.modeling_electra import ElectraModel
from transformers import ElectraModel
import os
train_k = pd.read_csv('submission_beike_0.8163005990197858.tsv', sep='\t', header=None)  # roberta
train_k.columns = ['id', 'id_sub', 'label']
a1 = train_k['label'].tolist()


def get_count(l):
    return l.count(1)


files = os.listdir('./esmble_result')
labels = []  # 存储voting前的结果
label = []  # 存储voting后的结果
for file in files:
    train = pd.read_csv(file, sep='\t', header=None)
    train.columns = ['id', 'id_sub', 'label']
    labels.append(train['label'].tolist())
c021 = 0
c120 = 0
for j in range(53757):
    each_predict = []
    for i in range(len(labels)):
        each_predict.append(labels[i][j])
    val = a1[j]

    if each_predict.count(1) == 8 and val == 0:
        # print(j+1)
        c021 += 1
        label.append(1)
    elif each_predict.count(0) == 8 and val == 1:
        print(j + 1)
        c120 += 1
        label.append(0)
    else:
        label.append(val)
    count1 = get_count(each_predict)
    # if count1 >= 5:
    #     label.append(1)
    # else:
    #     label.append(0)
# train = pd.read_csv('submission_beike_0.7999278694436931.tsv', sep='\t', header=None)  # ERNIE
# train.columns = ['id', 'id_sub', 'label']

# train_k1 = pd.read_csv('submission_beike_0.7946565574510334.tsv', sep='\t', header=None)  # bert base + pooling +对抗
# train_k1.columns = ['id', 'id_sub', 'label']
# train_k2 = pd.read_csv('submission_beike_0.774041511828152.tsv', sep='\t', header=None)
# train_k2.columns = ['id', 'id_sub', 'label']
# train_k3 = pd.read_csv('submission_beike_0.7816313987180613.csv', sep='\t', header=None)  # baseline
# train_k3.columns = ['id', 'id_sub', 'label']
# train_k4 = pd.read_csv('submission_beike_120021.tsv', sep='\t', header=None) # bert_base 1
# train_k4.columns = ['id', 'id_sub', 'label']
#
test_left = pd.read_csv('./test/test.query.tsv', sep='\t', header=None, encoding='gbk')
test_left.columns = ['id', 'q1']
test_right = pd.read_csv('./test/test.reply.tsv', sep='\t', header=None, encoding='gbk')
test_right.columns = ['id', 'id_sub', 'q2']
df_test = test_left.merge(test_right, how='left')
df_test.to_csv('submission_beike_{}.tsv'.format('look'), index=False, header=None,
                                              sep='\t')
# train_left = pd.read_csv('./train/train.query.tsv', sep='\t', header=None)
# train_left.columns = ['id', 'q1']
# train_right = pd.read_csv('./train/train.reply.tsv', sep='\t', header=None)
# train_right.columns = ['id', 'id_sub', 'q2', 'label']
# df_train = train_left.merge(train_right, how='left')
# df_train.to_csv('submission_beike_{}.tsv'.format('trainlook'), index=False, header=None,
#                                               sep='\t')
# test_ques = dict()
# all_ques = df_test['q1']
# all_reply = df_test['q2']
# label = []
#
# a = train['label'].tolist()
# a2 = train_k1['label'].tolist()
# a3 = train_k2['label'].tolist()
# a4 = train_k3['label'].tolist()
# a5 = train_k4['label'].tolist()
# print(a2.count(1))
# print(a.count(1))
# c = 0
# c5 = 0
# nc = 0
# nc5 = 0
#
#
# def get_count(val, e1, e2, e3, e4, e5):
#     count = 0
#     if e1 == val:
#         count+=1
#     if e2 == val:
#         count+=1
#     if e3 == val:
#         count+=1
#     if e4 == val:
#         count+=1
#     if e5 == val:
#         count+=1
#     return count
#
#
# label_a = []
# for i in range(len(a)):
#     if a[i] != a1[i]:
#         print(i+1)
#         nc+=1
    # if a[i] == 0:
    #     c5+=1
    # else:
    #     nc5+=1
    # label_a.append(a[i])
    # if get_count(0, a1[i], a2[i], a3[i], a4[i], a5[i]) == 5 and a[i] == 1:
    #     label.append(0)
    #     nc += 1
    # # elif a2[i]==0 and a[i]==1 and a1[i] == 1 and a3[i]==1 and a4[i]==1:
    # #     c+=1
    # #     label.append(1)
    # elif get_count(1, a1[i], a2[i], a3[i], a4[i], a5[i]) == 5 and a[i] == 0:
    #     # print(i + 1)
    #     label.append(1)
    #     c += 1
    # else:
    #     label.append(a[i])
    # else:
    #     c += 1
    # label.append(1)
# print(nc)
# print(c)
df_test['label'] = np.asarray(label).astype(int)
df_test[['id', 'id_sub', 'label']].to_csv('submission_beike_{}.tsv'.format('10vote1'), index=False,
                                          header=None, sep='\t')
