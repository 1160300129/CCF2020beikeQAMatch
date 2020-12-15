# coding: UTF-8
# @Author : szbw
# @Email : 20S003015@stu.hit.edu.cn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, FGM, PGD, FocalLoss
from tqdm import tqdm
import math
import transformers
from torch.optim import Adam, swa_utils
# from pytorch_pretrained_bert.optimization import BertAdam
from torch.optim.lr_scheduler import LambdaLR,CosineAnnealingLR


def get_cycle_schedule(optimizer, cycle_steps, last_epoch=-1):
    def lr_lambda(current_step):
        return max(0.08, 1 - (current_step % cycle_steps)/cycle_steps)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(config, model, train_iter, dev_iter, fold=None):
    start_time = time.time()
    model.train()
    # model.load_state_dict()
    param_optimizer = list(model.named_parameters())
    warmup_steps = math.ceil(len(train_iter.batches) * config.num_epochs / config.batch_size * 0.1)
    steps_per_epoch = len(train_iter)

    num_train_steps = int(steps_per_epoch * config.num_epochs)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_params = {'lr': config.learning_rate, 'eps': 1e-6, 'correct_bias': False}
    optimizer = transformers.optimization.AdamW(optimizer_grouped_parameters, **optimizer_params)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    # model.train()
    fgm = FGM(model)
    # pgd = PGD(model)
    # K=3
    # criterion = FocalLoss(gamma=0)
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            # model.zero_grad()
            outputs = model(trains)
            loss = F.cross_entropy(outputs, labels)
            # loss = F.binary_cross_entropy_with_logits(outputs, labels.type_as(outputs))
            # loss = F.binary_cross_entropy(outputs, labels.float())
            loss.backward()

            # pgd.backup_grad()
            # # 对抗训练
            # for t in range(K):
            #     pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
            #     if t != K - 1:
            #         model.zero_grad()
            #     else:
            #         pgd.restore_grad()
            #     outputs = model(trains)
            #     loss_adv = F.cross_entropy(outputs, labels)
            #     loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            # pgd.restore()  # 恢复embedding参数
            # 梯度下降，更新参数

            fgm.attack()  # 在embedding上添加对抗扰动
            outputs = model(trains)
            loss_adv = F.cross_entropy(outputs, labels, weight=torch.FloatTensor([3, 1], dtype=float, device='cuda'))

            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if total_batch % 200 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                a11 = 0
                for i in range(len(true)):
                    if true[i] == 1 and predic[i] == 1:
                        a11 += 1
                p = a11 / list(predic).count(1) if list(predic).count(1) != 0 else 0
                r = a11 / list(true).count(1) if list(true).count(1) != 0 else 0
                f1 = (2 * p * r) / (p + r) if p + r != 0 else 0
                if total_batch % 400 == 0:
                    dev_f1, dev_p, dev_r, dev_loss = evaluate(config, model, dev_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        # modules.save(config.save_path.split('/')[0] + '/best_' + str(fold) + config.save_path.split('/')[1].split('.')[0])
                        torch.save(model.state_dict(),
                                   config.save_path.split('/')[0] + '/best_' + str(fold) + config.save_path.split('/')[1])

                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train f1: {2:>6.2%},  Val Loss: {3:>5.2},  Val p: {4:>6.2%},   Val r: {5:>6.2%},   Val f1: {6:>6.2%},  Time: {7} {8}'
                    print(msg.format(total_batch, loss_adv.item(), f1, dev_loss, dev_p, dev_r, dev_f1, time_dif, improve))
                else:
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train f1: {2:>6.2%}'
                    print(msg.format(total_batch, loss_adv.item(), f1))
                model.train()
            total_batch += 1

            # if total_batch - last_improve > config.require_improvement:
            #     # 验证集loss超过1000batch没下降，结束训练
            #     print("No optimization for a long time, auto-stopping...")
            #     flag = True
            #     break
        # if flag:
        #     break
        # if epoch % 3 == 2:
        #     swa_model.update_parameters(model)
        #     evaluate(config, swa_model.module, )
            # torch.save(swa_model.state_dict(), config.save_path.split('/')[0] + '/swa_' + str(fold) + config.save_path.split('/')[1])
        if fold is None:
            torch.save(model.state_dict(), config.save_path)
        else:
            torch.save(model.state_dict(),
                       config.save_path.split('/')[0] + '/' + str(fold) + config.save_path.split('/')[1])
    # test(config, model, test_iter)


def predict(config, model, data_iter, fold, activation='sigmoid'):
    model.eval()
    # if fold == 4:
    #     fold = 3
    model.load_state_dict(torch.load(config.save_path))
    # model.load_state_dict(torch.load(config.save_path.split('/')[0] + '/' + str(fold) + config.save_path.split('/')[1]))
    predict_all = []
    with torch.no_grad():
        for texts, labels in tqdm(data_iter):
            output = model(texts)
            if activation == 'sigmoid':
                prob = torch.sigmoid(output)
            if activation == 'softmax':
                prob = torch.softmax(output, dim=1)
            u = prob.data.cpu().numpy()[:, 1]
            u = u.reshape(output.shape[0], 1)
            for each in u:
                predict_all.append(each)
    return np.asarray(predict_all)



def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            if not test:
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            # e+=1

    a11 = 0
    for i in range(len(labels_all)):
        if labels_all[i] == 1 and predict_all[i] == 1:
            a11 += 1
    p = a11 / list(predict_all).count(1) if list(predict_all).count(1) != 0 else 0
    r = a11 / list(labels_all).count(1)
    f1 = (2 * p * r) / (p + r) if p + r != 0 else 0
    # report = metrics.classification_report(labels_all, predict_all, target_names=['0', '1'], digits=4)
    # print(report)
    return f1, p, r, loss_total / len(data_iter)