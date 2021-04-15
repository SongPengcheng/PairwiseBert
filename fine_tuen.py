#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 11:05 上午
# @Author  : SPC
# @FileName: fine_tuen.py
# @Software: PyCharm
# @Desn    ：
import pbert_argparse
from SemDataset import PairwiseSemSimDataset
from trainer import Trainer
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger.info("Training Bert with Pairwise Strategy")
parse = pbert_argparse.parse
args = parse.parse_args()
args.data_dir = "data"
args.model_name_or_path = "output/checkpoint-9000"
args.train_batch_size = 64
args.dev_batch_size = 8
args.learning_rate = 2e-5
args.task_name = "sim"
args.do_train = True
args.do_eval = False
args.do_predict = False
args.save_steps = 1000
args.output_dir = "output"
args.save_total_limit = 3
args.num_train_epochs = 3
train_dataset = PairwiseSemSimDataset(
    data_path=args.data_dir,
    pretrain_path=args.model_name_or_path,
    max_seq_length=args.max_seq_length,
    mode="train"
)
from transformers import BertForSequenceClassification, BertConfig
bert_config = BertConfig.from_pretrained(args.model_name_or_path)
model = BertForSequenceClassification.from_pretrained(args.model_name_or_path,config=bert_config)
trainer = Trainer(args=args, model=model, train_dataset=train_dataset)
trainer.train(args.model_name_or_path)
