#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/12 5:11 下午
# @Author  : SPC
# @FileName: pbert_argparse.py
# @Software: PyCharm
# @Desn    ：

import argparse
parse = argparse.ArgumentParser(description="arguments for bert model")

ModelArgsGroup = parse.add_argument_group(title="model_args")
DataArgsGroup = parse.add_argument_group(title="data_args")
TrainArgsGroup = parse.add_argument_group(title="train_args")

ModelArgsGroup.add_argument(
    '--model_name_or_path', type=str, default=None,
    help="Path to pretrained model or model identifier from huggingface.co/models"
)
ModelArgsGroup.add_argument(
    '--config_name', type=str, default=None,
    help="Pretrained config name or path if not the same as model_name"
)
ModelArgsGroup.add_argument(
    '--tokenizer_name', type=str, default=None,
    help="Pretrained tokenizer name or path if not the same as model_name"
)
ModelArgsGroup.add_argument(
    '--cache_dir', type=str, default=None,
    help="Where do you want to store the pretrained models downloaded from s3"
)

DataArgsGroup.add_argument(
    '--task_name', type=str, default=None,
    help="The name of the task to train on"
)
DataArgsGroup.add_argument(
    '--data_dir', type=str, default=None,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task."
)
DataArgsGroup.add_argument(
    '--max_seq_length', type=int, default=128,
    help= "The maximum total input sequence length after tokenization. Sequences longer "
          "than this will be truncated, sequences shorter will be padded."
)
DataArgsGroup.add_argument(
    '--overwrite_cache', type=bool, default=False,
    help="Overwrite the cached training and evaluation sets"
)

TrainArgsGroup.add_argument(
    '--output_dir', type=str, default=None
)
TrainArgsGroup.add_argument(
    '--overwrite_output_dir', type=bool, default=False
)
TrainArgsGroup.add_argument(
    '--do_train', type=bool, default=False
)
TrainArgsGroup.add_argument(
    '--do_eval', type=bool, default=False
)
TrainArgsGroup.add_argument(
    '--do_predict', type=bool, default=False
)
TrainArgsGroup.add_argument(
    '--evaluate_during_training', type=bool, default=False
)
TrainArgsGroup.add_argument(
    '--train_batch_size', type=int, default=32
)
TrainArgsGroup.add_argument(
    '--eval_batch_size', type=int, default=8
)
TrainArgsGroup.add_argument(
    '--per_device_train_batch_size', type=int, default=8
)
TrainArgsGroup.add_argument(
    '--per_device_eval_batch_size', type=int, default=8
)
TrainArgsGroup.add_argument(
    '--per_gpu_train_batch_size', type=int, default=None,
    help="Deprecated, the use of `--per_device_train_batch_size` is preferred. "
         "Batch size per GPU/TPU core/CPU for training."
)
TrainArgsGroup.add_argument(
    '--per_gpu_eval_batch_size', type=int, default=None,
    help="Deprecated, the use of `--per_device_eval_batch_size` is preferred."
         "Batch size per GPU/TPU core/CPU for evaluation."
)
TrainArgsGroup.add_argument(
    '--gradient_accumulation_steps', type=int, default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass."
)
TrainArgsGroup.add_argument(
    '--learning_rate', type=float, default=5e-5
)
TrainArgsGroup.add_argument(
    '--weight_decay', type=float, default=0.0
)
TrainArgsGroup.add_argument(
    '--adam_epsilon', type=float, default=1e-8
)
TrainArgsGroup.add_argument(
    '--max_grad_norm', type=float, default=1.0
)
TrainArgsGroup.add_argument(
    '--num_train_epochs', type=float, default=3.0
)
TrainArgsGroup.add_argument(
    '--max_steps', type=int, default=1
)
TrainArgsGroup.add_argument(
    '--warmup_steps', type=int, default=0
)
TrainArgsGroup.add_argument(
    '--logging_dir', type=str, default=None
)
TrainArgsGroup.add_argument(
    '--logging_first_step', type=bool, default=False
)
TrainArgsGroup.add_argument(
    '--logging_steps', type=int, default=500
)
TrainArgsGroup.add_argument(
    '--save_steps', type=int, default=500
)
TrainArgsGroup.add_argument(
    '--save_total_limit', type=int, default=None
)
TrainArgsGroup.add_argument(
    '--no_cuda', type=bool, default=False
)
TrainArgsGroup.add_argument(
    '--seed', type=int, default=42
)
TrainArgsGroup.add_argument(
    '--fp16', type=bool, default=False, help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
)
TrainArgsGroup.add_argument(
    '--fp16_opt_level', type=str, default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
         "See details at https://nvidia.github.io/apex/amp.html"
)
TrainArgsGroup.add_argument(
    '--local_rank', type=int, default="-1",
    help="For distributed training: local_rank"
)
TrainArgsGroup.add_argument(
    '--tpu_num_cores', type=int, default=None,
    help="TPU: Number of TPU cores (automatically passed by launcher script)"
)
TrainArgsGroup.add_argument(
    '--tpu_metrics_debug', type=bool, default=False,
    help="TPU: Whether to print debug metrics"
)