#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/13 10:34 上午
# @Author  : SPC
# @FileName: SemDataset.py
# @Software: PyCharm
# @Desn    ：

import argparse
import numpy as np
import math
from torch.utils.data.dataset import Dataset
from SemProcessor import glue_convert_examples_to_features, pairwise_convert_examples_to_features, SemSimProcessor, PairwiseSemSimProcessor
from typing import Optional
from transformers import BertTokenizer
from transformers.data.processors.utils import InputFeatures

class SemSimDataset(Dataset):
    def __init__(
            self,
            args: argparse.ArgumentParser,
            tokenizer: BertTokenizer = None,
            limit_length: Optional[int] = None,
            mode: str = "train"
    ):
        super(SemSimDataset,self).__init__()
        self.args = args
        self.data_path = self.args.data_dir
        self.tokenizer = tokenizer
        self.processor = SemSimProcessor()
        self.output_mode = "classification"
        self.label_list = self.processor.get_labels()

        if mode == "eval":
            examples = self.processor.get_dev_examples(self.data_path)
        elif mode == "test":
            examples = self.processor.get_test_examples(self.data_path)
        else:
            examples = self.processor.get_train_examples(self.data_path)
        if limit_length is not None:
            examples = examples[:limit_length]
        self.features = glue_convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.args.max_seq_length,
            label_list=self.label_list,
            output_mode=self.output_mode,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

class PairwiseSemSimDataset(Dataset):
    def __init__(
            self,
            args: argparse.ArgumentParser = None,
            tokenizer: BertTokenizer = None,
            limit_length: Optional[int] = None,
            mode: str = "train"
    ):
        super(PairwiseSemSimDataset, self).__init__()
        self.args = args
        self.data_path = self.args.data_dir
        self.processor = PairwiseSemSimProcessor()
        self.output_mode = "classification"
        self.tokenizer = tokenizer
        self.label_list = self.processor.get_labels()
        self.max_seq_length = args.max_seq_length
        self.limit_length = limit_length
        self.features = self.getFeatures(mode)
        self.X1, self.X2, self.Y, self.weight = self._transform_pairwise()

    def getFeatures(self,mode):
        if mode == "dev":
            examples = self.processor.get_dev_examples(self.data_path)
        elif mode == "test":
            examples = self.processor.get_test_examples(self.data_path)
        else:
            examples = self.processor.get_train_examples(self.data_path)
        if self.limit_length is not None:
            examples = examples[:self.limit_length]
        return pairwise_convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.max_seq_length,
            label_list=self.label_list,
            output_mode=self.output_mode,
        )

    @staticmethod
    def _CalcDCG(labels):
        sumdcg = 0.0
        for i in range(len(labels)):
            rel = labels[i]
            if rel != 0:
                sumdcg += ((2 ** rel) - 1) / math.log2(i + 2)
        return sumdcg

    def _fetch_qid_data(self, eval_at=None):
        qid = []
        rel = []
        for feature in self.features:
            qid.append(int(feature[0]))
            rel.append(int(feature[2]))
        qid = np.asarray(qid)
        rel = np.asarray(rel)
        qid_unique, qid2indices, qid_inverse_indices = np.unique(qid, return_index=True, return_inverse=True)
        # qid2rel长度等于qid_unique长度，即问题的数量
        qid2rel = [[] for _ in range(len(qid_unique))]
        for i, qid_unique_index in enumerate(qid_inverse_indices):
            qid2rel[qid_unique_index].append(rel[i])
        if eval_at:
            qid2dcg = [self._CalcDCG(qid2rel[i][:eval_at]) for i in range(len(qid_unique))]
            qid2idcg = [self._CalcDCG(sorted(qid2rel[i], reverse=True)[:eval_at]) for i in range(len(qid_unique))]
        else:
            qid2dcg = [self._CalcDCG(qid2rel[i]) for i in range(len(qid_unique))]
            qid2idcg = [self._CalcDCG(sorted(qid2rel[i], reverse=True)) for i in range(len(qid_unique))]
        return qid2indices, qid2rel, qid2idcg, qid2dcg

    def _transform_pairwise(self):
        qid2indices, qid2rel, qid2idcg, _ = self._fetch_qid_data()
        Features1 = []
        Features2 = []
        weight = []
        Y = []
        for qid_unique_idx in range(len(qid2indices)):
            if qid2idcg[qid_unique_idx] == 0:
                continue
            IDCG = 1.0 / qid2idcg[qid_unique_idx]
            rel_list = qid2rel[qid_unique_idx]
            qid_start_idx = qid2indices[qid_unique_idx]
            for pos_idx in range(len(rel_list)):
                for neg_idx in range(len(rel_list)):
                    if rel_list[pos_idx] <= rel_list[neg_idx]:
                        continue
                    # calculate lambda
                    pos_loginv = 1.0 / math.log2(pos_idx + 2)
                    neg_loginv = 1.0 / math.log2(neg_idx + 2)
                    pos_label = rel_list[pos_idx]
                    neg_label = rel_list[neg_idx]
                    original = ((1 << pos_label) - 1) * pos_loginv + ((1 << neg_label) - 1) * neg_loginv
                    changed = ((1 << neg_label) - 1) * pos_loginv + ((1 << pos_label) - 1) * neg_loginv
                    delta = (original - changed) * IDCG
                    if delta < 0:
                        delta = -delta
                    # balanced class
                    if 1 != (-1) ** (qid_unique_idx + pos_idx + neg_idx):
                        Features1.append(self.features[qid_start_idx + pos_idx][1])
                        Features2.append(self.features[qid_start_idx + neg_idx][1])
                        weight.append(delta)
                        Y.append(1)
                    else:
                        Features1.append(self.features[qid_start_idx + neg_idx][1])
                        Features2.append(self.features[qid_start_idx + pos_idx][1])
                        weight.append(delta)
                        Y.append(0)
        return Features1, Features2, Y, weight

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X1[item],self.X2[item],self.Y[item],self.weight[item]