# coding: UTF-8
# @Author : szbw
# @Email : 20S003015@stu.hit.edu.cn
import time
import torch
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from train_eval import train, predict
from importlib import import_module
import pandas as pd
from config import Config
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.utils import shuffle
from utils import build_dataset, build_iterator, get_time_dif


def load_dataset(df, columns, pad_size=100):
    """
    对输入数据做处理，获取bert输入必要的信息
    :param df: dataframe格式数据
    :param columns: 选取的行
    :param pad_size: 填充至统一句长的值
    :return: List[tuple[]]
    """
    contents = []

    for _, instance in tqdm(df[columns].iterrows()):
        if 'label' in columns:
            content1, content2, label = instance.q1, instance.q2, instance.label
        else:
            content1, content2 = instance.q1, instance.q2
        id_sub = instance.id_sub
        content2 = '#'+str(id_sub)+content2
        inputs = config.tokenizer.encode_plus(
            content1, content2,
            add_special_tokens=True,
            max_length=pad_size,
            truncation_strategy='longest_first',
            truncation=True
        )
        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = pad_size - len(input_ids)
        padding_id = 0
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        if 'label' in columns:
            contents.append((input_ids, int(label), input_masks.count(1), input_masks, input_segments))
        else:
            contents.append((input_ids, -1, input_masks.count(1), input_masks, input_segments))
    return contents


def search_f1(y_true, y_pred):
    """
    动态调整阈值
    :param y_true:真实标签
    :param y_pred: 预测标签
    :return: 最高的f1值和最好的阈值
    """
    best = 0
    best_t = 0
    for i in range(30, 80):
        tres = i / 100
        y_pred_bin = (y_pred > tres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best:
            best = score
            best_t = tres
    print('best', best)
    print('thres', best_t)
    return best, best_t


if __name__ == '__main__':
    model_name = 'bert_nezhabaseline'  # .py文件的名称 里面是模型
    x = import_module(model_name)
    config = Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    input_categories = ['q1', 'q2', 'label', 'id_sub']
    test_categories = ['q1', 'q2', 'id_sub']
    train_left = pd.read_csv('./train/train.query.tsv', sep='\t', header=None)
    train_left.columns = ['id', 'q1']
    train_right = pd.read_csv('./train/train.reply.tsv', sep='\t', header=None)
    train_right.columns = ['id', 'id_sub', 'q2', 'label']
    df_train = train_left.merge(train_right, how='left')[:100]
    df_train['q2'] = df_train['q2'].fillna('null')
    test_left = pd.read_csv('./test/test.query.tsv', sep='\t', header=None, encoding='gbk')
    test_left.columns = ['id', 'q1']
    test_right = pd.read_csv('./test/test.reply.tsv', sep='\t', header=None, encoding='gbk')
    test_right.columns = ['id', 'id_sub', 'q2']
    df_test = test_left.merge(test_right, how='left')[:10]
    # df_train = shuffle(df_train)
    # with open('add_test_to_train', 'r', encoding='utf8') as f:
    #     b = f.read()
    #     lines = b.strip().split('\n')
    #     z = 1
    #     datas = pd.DataFrame()
    #     for line in lines:
    #         if z % 5000 == 0:
    #             print(z)
    #         data = df_test.iloc[int(line)]
    #         series = pd.Series(
    #             {"id": data.id, "q1": data.q1, "id_sub": data.id_sub, "q2": data.q2, "label": 1 if z <= 8669 else 0})
    #         datas = datas.append(series, ignore_index=True)
    #
    #         # df_train.loc[df_train.shape[0]]
    #         z += 1
    #     df_train = df_train.append(datas, ignore_index=True)

    start_time = time.time()
    datas = load_dataset(df_train, input_categories, config.pad_size)
    test_inputs = load_dataset(df_test, test_categories, config.pad_size)
    gkf = GroupKFold(n_splits=5).split(X=df_train.q2, groups=df_train.id)

    valid_preds = []
    test_preds = []
    print("一共"+str(df_train.shape[0])+"个训练语句")
    oof = np.zeros((len(df_train), 1))
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        # if fold<10:
        #     continue
        model = x.Model(config).to(config.device)
        print("Loading " + str(fold + 1) + " fold data...")
        train_idx = shuffle(train_idx)
        train_inputs = [datas[i] for i in train_idx]
        valid_inputs = [datas[i] for i in valid_idx]
        train_iter = build_iterator(train_inputs, config.batch_size, config)
        dev_iter = build_iterator(valid_inputs, config.test_batch, config)
        test_iter = build_iterator(test_inputs, config.test_batch, config)
        valid_outputs = np.array([], dtype=int)
        for d, (text, labels) in enumerate(dev_iter):
            valid_outputs = np.append(valid_outputs, labels.data.cpu().numpy())
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        train(config, model, train_iter, dev_iter, fold)
        oof_p = predict(config, model, dev_iter, fold, activation='softmax')
        oof[valid_idx] = oof_p
        valid_preds.append(oof_p)

        f1, t = search_f1(valid_outputs, valid_preds[-1])
        print('validation score = ', f1)
        each_fold_predict = predict(config, model, test_iter, fold, activation='softmax')
        test_preds.append(each_fold_predict)
        sub = each_fold_predict > t
        # df_test['label'] = sub.astype(int)
        # df_test[['id', 'id_sub', 'label']].to_csv('submission_beike_{}.tsv'.format(fold), index=False,
        #                                           header=None, sep='\t')
        torch.cuda.empty_cache()

    outputs = np.asarray(df_train['label'])
    best_score, best_t = search_f1(outputs, oof)

    sub = np.average(test_preds, axis=0)
    sub = sub > best_t
    df_test['label'] = sub.astype(int)
    df_test[['id', 'id_sub', 'label']].to_csv('submission_beike_{}.tsv'.format(best_score), index=False, header=None,
                                              sep='\t')
    # test_data = build_dataset(config, train=False, test=True)
    # test_iter = build_iterator(test_data, config)
    # torch.save(model.state_dict(), 'THUCNews/saved_dict/test.ckpt')
    # predict_all = test_k(config, test_iter)
    # predict_all = predict_all.tolist()
    # with open('submissionk.tsv', 'w', encoding='utf8') as f, open('test/test.reply.tsv', 'r', encoding='gbk') as f1:
    #     b = f1.read()
    #     lines = b.strip().split('\n')
    #     for i in range(len(lines)):
    #         f.write(lines[i].strip().split('\t')[0] + '\t' + lines[i].strip().split('\t')[1]+'\t'+str(predict_all[i])+'\n')

    # print(predict_all)
