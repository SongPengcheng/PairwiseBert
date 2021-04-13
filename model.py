#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/12 10:35 上午
# @Author  : SPC
# @FileName: model.py.py
# @Software: PyCharm
# @Desn    ：

import os
import torch
import argparse
import pbert_argparse
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim
from transformers import BertForSequenceClassification, BertConfig
from transformers.data.data_collator import DataCollator, DefaultDataCollator
from SemDataset import SemSimDataset, PairwiseSemSimDataset
from typing import Optional, Union, Any, Dict, List, NewType, Tuple
from tqdm.auto import tqdm, trange
from dataclasses import dataclass
os.environ['CUDA_VISIBLE_DEVICES']='0'
InputDataClass = NewType("InputDataClass", Any)

@dataclass
class PairwiseDataCollator(DataCollator):

    def collate_batch(self, data_set):
        def make_feature_batch(dataset, idx) -> Dict[str, torch.Tensor]:
            first = dataset[0][idx]
            if hasattr(first, "label") and first.label is not None:
                if type(first.label) is int:
                    labels = torch.tensor([d[idx].label for d in dataset], dtype=torch.long)
                else:
                    labels = torch.tensor([d[idx].label for d in dataset], dtype=torch.float)
                batch = {"labels": labels}
            elif hasattr(first, "label_ids") and first.label_ids is not None:
                if type(first.label_ids[0]) is int:
                    labels = torch.tensor([d[idx].label_ids for d in dataset], dtype=torch.long)
                else:
                    labels = torch.tensor([d[idx].label_ids for d in dataset], dtype=torch.float)
                batch = {"labels": labels}
            else:
                batch = {}
            for k, v in vars(first).items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    batch[k] = torch.tensor([getattr(d[idx], k) for d in dataset], dtype=torch.long)
            return batch

        def make_data_batch(dataset, idx):
            first = dataset[0][idx]
            if type(first) is int:
                data_batch = torch.tensor([item[idx] for item in dataset], dtype=torch.long)
            else:
                data_batch = torch.tensor([item[idx] for item in dataset], dtype=torch.float)

        return make_feature_batch(data_set, 0), make_feature_batch(data_set, 1), make_data_batch(data_set,
                                                                                                 2), make_data_batch(
            data_set, 3)


class SemSimModel(nn.Module):
    def __init__(self, model_path, out_dim=1, dropout=0.1):
        super(SemSimModel, self).__init__()
        self.bert_config = BertConfig.from_pretrained(model_path + 'config.json')
        self.bert = BertForSequenceClassification.from_pretrained(model_path + 'pytorch_model.bin',
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


class Trainer(nn.Module):
    def __init__(
            self,
            args: argparse.ArgumentParser,
            model: SemSimModel = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,

    ):
        super(Trainer, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DefaultDataCollator()

    def forward(self, input_1, input_2):
        # dx = 1 if not input_1["labels"] else 0
        input_1_property = self.model(**input_1)[1][:, 1]
        input_2_property = self.model(**input_2)[1][:, 1]
        output = torch.sigmoid((input_1_property - input_2_property))
        return output

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(self.train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(self.train_dataset)
        )
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )
        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )
        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        sampler = SequentialSampler(test_dataset)
        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )
        return data_loader

    def get_pairwise_dataloader(self, pairwise_dataset: Dataset, batchsize: int) -> DataLoader:
        # 对于进行pairwise训练的数据，不论train，eval还是test都顺序进行采样
        sampler = SequentialSampler(pairwise_dataset)
        data_collator = PairwiseDataCollator()
        data_loader = DataLoader(
            dataset=pairwise_dataset,
            sampler=sampler,
            batch_size=batchsize,
            collate_fn=data_collator.collate_batch,
        )
        return data_loader

    def train(self):
        dataset = self.train_dataset
        dataloader = self.get_pairwise_dataloader(
            pairwise_dataset=dataset,
            batchsize=self.args.train_batch_size
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        for epoch in range(int(self.args.num_train_epochs)):
            for feature1, feature2, y, weight in tqdm(dataloader, desc="train"):
                optimizer.zero_grad()
                for k, v in feature1.items():
                    feature1[k] = v.to(trainer.device)
                for k, v in feature2.items():
                    feature2[k] = v.to(trainer.device)
                output = trainer(feature1, feature2)
                y.unsqueeze(1)
                weight.unsqueeze(1)
                criteran = nn.BCELoss(weight)
                loss = criteran(output, y)
                loss.backward()
                optimizer.step()

    def eval(self):
        pass

    def predict(self, dataloader: DataLoader):
        model = self.model
        for inputs in tqdm(dataloader, desc="predict"):
            # has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
        return outputs


parse = pbert_argparse.parse
args = parse.parse_args()
args.data_dir = "../data/ModelDataset/PathRank/qp_pairs/Pairwise/"
args.model_name_or_path = "../model/PathRank/bert_path_rank/binary_class/"
args.eval_batch_size = 2
test_dataset = PairwiseSemSimDataset(
    data_path=args.data_dir,
    pretrain_path=args.model_name_or_path,
    max_seq_length=args.max_seq_length,
    mode="train"
)

model = SemSimModel(model_path=args.model_name_or_path)
trainer = Trainer(args=args, model=model, train_dataset=test_dataset)
trainer.train()