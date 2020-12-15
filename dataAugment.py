# @Author : szbw
# @Email : 20S003015@stu.hit.edu.cn
from nlpcda import Simbert
import pandas as pd
from tqdm import tqdm
import jieba
import synonyms
import random
from back_translate import back_translate
from random import shuffle


random.seed(2019)

#停用词列表，默认使用哈工大停用词表
f = open('../eda_nlp_for_Chinese/stopwords/HIT_stop_words.txt', 'r', encoding='utf8')
stop_words = list()
for stop_word in f.readlines():
    stop_words.append(stop_word[:-1])


#考虑到与英文的不同，暂时搁置
#文本清理
'''
import re
def get_only_chars(line):
    #1.清除所有的数字
'''


########################################################################
# 同义词替换
# 替换一个语句中的n个单词为其同义词
########################################################################
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    return synonyms.nearby(word)[0]


########################################################################
# 随机插入
# 随机在语句中插入n个词
########################################################################
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

########################################################################
# 随机删除
# 以概率p删除语句中的词
########################################################################
def random_deletion(words, p):

    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words


def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4)
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    # print(words, "\n")

    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(''.join(a_words))

    # 随机插入ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(''.join(a_words))

    # 随机交换rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(''.join(a_words))

    # 随机删除rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(''.join(a_words))

    # print(augmented_sentences)

    return augmented_sentences
'''
构造标签为1的答案 输出为tttt进行数据增强
'''
# config = {
#         'model_path': './chinese_simbert_L-12_H-768_A-12',
#         'device': 'cuda',
#         'max_len': 100,
#         'seed': 1
# }
# simbert = Simbert(config=config)
# train_right = pd.read_csv('./train/train.reply.tsv', sep='\t', header=None)[:10]
# train_right.columns = ['id', 'id_sub', 'q2', 'label']
# ids = train_right['id'].tolist()
# q1 = train_right['q2'].tolist()
# label = train_right['label'].tolist()
# id_sub = train_right['id_sub'].tolist()
train_left = pd.read_csv('./train/train.query.tsv', sep='\t', header=None)
# train_left = pd.read_csv('./test/test.query.tsv', sep='\t', header=None, encoding='gbk')
train_left.columns = ['id', 'q1']
ids = train_left['id'].to_list()
q1 = train_left['q1'].to_list()
alpha = 0.1
num_aug = 4
new_idx = []
new_q1 = []
import random
for i in tqdm(range(len(ids))):
    '''
      首先 基于规则 增强和扩展问句 
    '''
    if len(q1[i]) <= 4:
        if '临街' in q1[i]:
            extend_query = '房子'+q1[i]
        elif '户型' in q1[i]:
            extend_query = '是什么'+q1[i]
        elif '光' in q1[i]:
            extend_query = '这个楼层'+q1[i]
        elif '看' in q1[i]:
            extend_query = '询问能不能看房：'+q1[i]
        elif '期' in q1[i]:
            extend_query = '这栋楼是几期的:'+q1[i]
        elif '税' in q1[i]:
            if '有' in q1[i]:
                extend_query = '询问有没有:'+q1[i]+'?'
            elif '多' in q1[i]:
                extend_query = '询问具体数字:'+q1[i]+'?'
            else:
                extend_query = q1[i]
        elif '小学' in q1[i]:
            extend_query = '询问小学地名:'+q1[i]
        elif '初中' in q1[i]:
            extend_query = '询问中学初中地名:'+q1[i]
        elif '满' in q1[i]:
            extend_query = '时间:房子现在满几年了?'
        elif '毛坯' in q1[i]:
            extend_query = '是不是毛坯:'+q1[i]
        else:
            extend_query = q1[i]
        new_idx.append(ids[i])
        new_q1.append(extend_query)
    else:
        sent = q1[i]
        '''
        规则 添加意图 替换一些地名
        '''
        # if '时候' in sent:
        #     extend_query = '具体时间:'+sent
        # elif '哪里' in sent or '在哪' in sent:
        #     extend_query = '地点:'+sent
        # elif '税' in sent or '利率' in sent:
        #     extend_query = '税或者利率几个点.'+sent
        # elif '卖' in sent:
        #     extend_query = '卖多少钱,价格多少:'+sent
        # elif '采光' in sent or '太阳' in sent:
        #     extend_query = '楼层采光:'+sent
        # elif '边户' in sent or '中间户' in sent or '边套' in sent or '中间套' in sent or '东户' in sent or '西户' in sent:
        #     extend_query = '是哪个位置的:'+sent
        # elif '位置' in sent or '靠' in sent or '临' in sent or '邻' in sent:
        #     extend_query = '是否:'+sent
        # elif '家具' in sent or '家电' in sent:
        #     extend_query = '家具家电:'+sent
        # elif '首付' in sent:
        #     extend_query = '几成,多少'+sent
        # elif '公积金' in sent:
        #     extend_query = '公积金:'+sent
        # elif '证' in sent or '本' in sent or '产权' in sent:
        #     extend_query = '房产证,产权:'+sent
        # elif '贷' in sent:
        #     if '可' in sent:
        #         extend_query = '能否可以:'+sent
        #     elif '年' in sent:
        #         extend_query = '贷款年限 时间:'+sent
        #     else:
        #         extend_query = sent
        # elif '学位' in sent or '校' in sent or '学区' in sent or '小学' in sent or '初中' in sent or '高中' in sent or '中学' in sent or '教育' in sent or '一小' in sent:
        #     if '学校' in sent:
        #         extend_query = '学校:'+sent
        #     elif '小学' in sent:
        #         extend_query = '小学:'+sent
        #     elif '初中' in sent:
        #         extend_query = '初中:' + sent
        #     elif '高中' in sent:
        #         extend_query = '高中:' + sent
        #     else:
        #         extend_query = sent
        # else:
        #     extend_query = sent
        new_q1.append(sent)
        new_idx.append(ids[i])
        aug_sentences = eda(sent, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        # if random.random()>0.5:
        #     back_translate_s = back_translate(sent)
        #     aug_sentences.append(back_translate_s)
        for s in aug_sentences:
            new_q1.append(s)
            new_idx.append(ids[i])


# new_q2 = []
# new_label = []
# new_id_sub = []
last_ids = 0
# for i in tqdm(range(len(ids))):
    # if label[i] == 1:
    #     new_id_sub.append(id_sub[i])

        # synonyms = simbert.replace(sent=sent, create_num=3)
        # for each in synonyms:
        #     print(each)
        #     new_idx.append(i)
        #     new_q1.append(each[0])

# c = {'id':new_idx,'id_sub':new_id_sub, 'q2':new_q1, 'label':new_label}
c = {'id':new_idx, 'q2':new_q1,}
df = pd.DataFrame(c)
df.to_csv('../ttt', index=False, header=None, sep='\t')

'''
将增强后的标签为1的数据 写入源数据
'''
# final_id = []
# final_sub_id = []
# final_q2 = []
# final_lable = []
# with open('train_augmented.txt', 'r', encoding='utf8') as f:
#     b = f.read()
#     arg_ids = {}
#     lines = b.strip().split('\n')
#     for line in lines:
#         each = line.strip().split('\t')
#         tmp_id = int(each[0].split(' ')[0])
#         tmp_sub_id = int(each[0].split(' ')[1])
#         if tmp_id in arg_ids.keys():
#             if tmp_sub_id in arg_ids[tmp_id].keys():
#                 arg_ids[tmp_id][tmp_sub_id].append(each[1])
#             else:
#                 arg_ids.setdefault(tmp_id, dict()).setdefault(tmp_sub_id, [each[1]])
#         else:
#             arg_ids.setdefault(tmp_id, dict()).setdefault(tmp_sub_id, [each[1]])
#     for i in range(len(ids)):
#         if label[i] == 1:
#             for arg_data in arg_ids[ids[i]][id_sub[i]]:
#                 final_id.append(ids[i])
#                 final_sub_id.append(id_sub[i])
#                 arg_data = ''.join(arg_data.split(' '))
#                 final_q2.append(arg_data)
#                 final_lable.append(1)
#         else:
#             final_id.append(ids[i])
#             final_sub_id.append(id_sub[i])
#             final_q2.append(q1[i])
#             final_lable.append(label[i])
#
# c = {'id': final_id, 'id_sub': final_sub_id, 'q2': final_q2, 'label': final_lable}
# df = pd.DataFrame(c)
# df.to_csv('../final_train.reply.tsv', index=False, header=None, sep='\t')
# f.close()
