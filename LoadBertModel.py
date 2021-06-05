#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 3:51 下午
# @Author  : SPC
# @FileName: LoadBertModel.py
# @Software: PyCharm
# @Desn    : 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 11:05 上午
# @Author  : SPC
# @FileName: fine_tune.py
# @Software: PyCharm
# @Desn    ：
import os
import pbert_argparse
import numpy as np
import dataclasses
from SemDataset import PairwiseSemSimDataset, SemSimDataset
from SemProcessor import tasks_num_labels, output_modes
from trainer import Trainer
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
parse = pbert_argparse.parse
args = parse.parse_args()
def LoadModel(model_path):
    args.model_name_or_path = model_path
    args.task_name = "sim"
    try:
        num_labels = tasks_num_labels[args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (args.task_name))
    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )
    predictor = Trainer(
        model=model,
        args=args,
        compute_metrics=None
    )
    return predictor, tokenizer
def PredictByTextPairs(predictor,tokenizer,text_pairs):
    test_dataset = SemSimDataset(args=args,tokenizer=tokenizer,text_pairs=text_pairs,mode="test")
    test_datasets =[test_dataset]
    for test_dataset in test_datasets:
        predictions = predictor.predict(test_dataset=test_dataset).predictions
        predictions = predictions[:, 1].tolist()
        return predictions

if __name__ == '__main__':
    model_path = "model"
    predictor,tokenizer = LoadModel(model_path)
    text_pairs = [("独立宣言的签署日期","独立宣言美国立国文书之一"),("证监会主席哪一年出生","证监会")]
    result = PredictByTextPairs(predictor,tokenizer,text_pairs)
    #print(result)




