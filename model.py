#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/12 10:35 上午
# @Author  : SPC
# @FileName: model.py.py
# @Software: PyCharm
# @Desn    ：

import pbert_argparse
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
from SemDataset import PairwiseSemSimDataset

class SemSimModel(nn.Module):
    def __init__(self, model_path, out_dim=1, dropout=0.1):
        super(SemSimModel, self).__init__()
        self.bert_config = BertConfig.from_pretrained(model_path)
        self.bert = BertForSequenceClassification.from_pretrained(model_path,
                                                                  config=self.bert_config)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            labels,
        )
        return outputs
