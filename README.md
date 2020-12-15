# CCF2020beikeQAMatch

2020CCF大数据与计算智能大赛 房地产行业聊天问答匹配赛题  比赛地址链接:https://www.datafountain.cn/competitions/474

最终成绩A榜32/2985；B榜24/2985

依赖包:加载预训练模型使用的transformers(pip install transformers 好像4.0发布了 需要改一丢丢地方 才能将3.x的代码跑通) 
深度学习框架使用的pytorch

项目结构：
   train:训练数据
   test:测试数据
   bert.py：最初的baseline，直接使用了nsp任务
   bert_baselines.py bert+poolig的方式 也是本次比赛的最终模型
   ---bert_nezhabaseline.py 哪吒模型 也采取了预训练+pooling的方式
   ---file_utils.py   nezha在bert的基础上魔改了一下 transformer无法加载 只能copy的华为的仓库 调用modeling_nezha里的函数      
   ---modeling_nezha.py  copy的华为的仓库
   config.py 学习率 batch 存储路径 预训练模型路径 在此修改
   dataAugment.py 数据增强主要使用了EDA 和 对问句进行了简单的扩展(最后实际没啥用，单纯的传上来)
   model_ensmble.py 模型集成 集成'路径xxx'（自己修改）下的文件， 具体集成方式n个投票1个
   run_kfold.py 5折划分及预测
   train_eval.py 训练和验证
   utils.py 


使用好使的trick
(1) 5折交叉验证
(2) bert后+pooling
(3) 对抗训练
(4) 模型融合
(5) 添加排序信息(id_sub)


暂时无效的trick:
(1) 将bert_base->large 效果跟base基本持平 而且训练时会玄学出现某一折全是f1值0的情况 但最后还是将roberta_large的结果融合到最终结果中
(2) 伪标签 在A榜效果不明显(20%的测试集和全部测试集都尝试了),可能由于A榜是部分测试集的因素
    但B榜的最终结果有提升，但不知道是否是伪标签提升的，因为b榜提交不给看成绩。
    (可以试一下将置信度大的加进去,例如5个baseline模型预测全为1)
(3)数据增强 EDA我测试没有效果的提升 甚至下降
(4) PET文本分类 看到苏剑林老师的一篇文章 自己尝试 效果巨差...（这就肯定是我写的代码的问题 因为我有同学尝试这种方法确实会有提升）
    top1 的博客也介绍了这个方法 坐等大佬！！


整个比赛的具体情况详细见ppt文件夹下的fxh.pptx 是我自己的一点点总结

如果对该项目的一些细节有更多的疑问 欢迎email我 842365309@qq.com

最后 顺手给个Star吧 谢谢您了！！！

