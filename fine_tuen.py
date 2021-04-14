#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 11:05 上午
# @Author  : SPC
# @FileName: fine_tuen.py
# @Software: PyCharm
# @Desn    ：
import pbert_argparse
from SemDataset import PairwiseSemSimDataset
from model import SemSimModel
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
args.model_name_or_path = "bert-base-chinese"
args.train_batch_size = 64
args.dev_batch_size = 8
args.learning_rate = 2e-5
args.task_name = "sim"
args.do_train = True
args.do_eval = False
args.do_predict = False
args.save_steps = 1000
args.output_dir = "output"
args.save_total_limit = 5
args.num_train_epochs = 3
train_dataset = PairwiseSemSimDataset(
    data_path=args.data_dir,
    pretrain_path=args.model_name_or_path,
    max_seq_length=args.max_seq_length,
    mode="train"
)
model = SemSimModel(model_path=args.model_name_or_path)
trainer = Trainer(args=args, model=model, train_dataset=train_dataset)
trainer.train()
