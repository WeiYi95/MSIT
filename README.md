# 古文无指导分词
自动分词指通过计算机技术手段，将由连续汉字字符构成的符号串进行分割，显式地呈现为由词组成的句子。对于古汉语自动分词任务，考虑到现有带标注文本较少，我们通过将非参数贝叶斯模型与预训练BERT模型相结合，提出无指导多阶段迭代训练（Multi-Stage Iterative Training，MSIT）分词框架，使用大量未标注文本进行无指导训练，提升模型的泛化能力。

# 配置
python == 3.6.8  
tensorflow-gpu == 1.13.1

# 使用语料
|数据集|字数|字表大小|内容来源|
|------|----|--------|--------|
|无指导训练语料|1.75千万|7151|《史藏》古文，主要包括《资治通鉴》、《史记》等|
《左传》小测试集|3万|1790|官方《左传》语料测试集|
《左传》大测试集|15万|3206|《左传》全部人工标注语料库|

由于版权原因，我们无法公开《左传》数据集。数据集请从官方网址（https://catalog.ldc.upenn.edu/LDC2017T14）获取。

# 模型训练
```
nohup python main.py > log
```
💾每阶段使用的语料全部在 data/msit.txt 中。可根据 log 中给出的信息，查看每一阶段具体使用的语料。

# 模型测试
```
python BERT_SEG.py --task_name="SEG" --do_train=False --do_eval=False --do_predict=True --data_dir=./ --vocab_file=./vocab.txt --bert_config_file=./bert_config.json --init_checkpoint=./stage_*_ckpt/model.ckpt --max_seq_length=24 --output_dir=./stage_*_ckpt
```
💾stage_*_ckpt：最优的模型参数在 best_ckpt.txt中。其中的模型参数均为最优候选。

# 预计结果
|Precision|Recall|F1|
|---------|------|--|
|0.92±0.01|0.94±0.01|0.93±0.01|  

# 参考
模型实现请参考（If you make use of this software for research purposes, we'll appreciate citing the following）：
```
俞敬松,魏一,张永伟,杨浩.基于非参数贝叶斯模型和深度学习的古文分词研究[J].中文信息学报,2020,39(6).
```
	
# 相关文献
[1] Jin Z, Tanakaishii K. Unsupervised Segmentation of Chinese Text by Use of Branching Entropy[C]. meeting of the association for computational linguistics, 2006: 428-435.  
[2] Magistry P, Sagot B. Unsupervized Word Segmentation: the Case for Mandarin Chinese[C]. meeting of the association for computational linguistics, 2012: 383-387.  
[3] Chen M, Chang B, Pei W, et al. A Joint Model for Unsupervised Chinese Word Segmentation[C]. em-pirical methods in natural language processing, 2014: 854-863.
