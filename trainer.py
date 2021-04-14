#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 11:02 上午
# @Author  : SPC
# @FileName: trainer.py
# @Software: PyCharm
# @Desn    ：
import os
import torch
import argparse
import json
import shutil
from model import SemSimModel
from packaging import version
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import logging
from transformers.data.data_collator import DataCollator, DefaultDataCollator
from transformers.optimization import AdamW,get_linear_schedule_with_warmup
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from typing import Optional, Union, Any, Dict, List, NewType, Tuple
from tqdm.auto import tqdm, trange
from dataclasses import dataclass
InputDataClass = NewType("InputDataClass", Any)
logger = logging.getLogger(__name__)

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
            data_batch = torch.tensor([item[idx] for item in dataset], dtype=torch.float)
            return data_batch
        return make_feature_batch(data_set, 0), make_feature_batch(data_set, 1), make_data_batch(data_set,2), make_data_batch(data_set, 3)

class Trainer(nn.Module):
    epoch: int
    global_step: int
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

    def get_optimizer(self,num_training_steps:int):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def _training_step(self, inputs, optimizer):
        feature1, feature2, y, weight = inputs
        for k, v in feature1.items():
            feature1[k] = v.to(self.device)
        for k, v in feature2.items():
            feature2[k] = v.to(self.device)
        input_1_property = self.model(**feature1)[1][:, 1]
        input_2_property = self.model(**feature2)[1][:, 1]
        output = torch.sigmoid((input_1_property - input_2_property))
        criteran = nn.BCELoss(weight.to(self.device).view(-1))
        loss = criteran(output.view(-1), y.to(self.device).view(-1))
        loss.backward()
        return loss.item()

    def is_local_master(self) -> bool:
        return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.tb_writer:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
        output = json.dumps({**logs, **{"step": self.global_step}})
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def train(self):
        train_dataset = self.train_dataset
        train_dataloader = self.get_pairwise_dataloader(
            pairwise_dataset=train_dataset,
            batchsize=self.args.train_batch_size
        )
        t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
        num_train_epochs = self.args.num_train_epochs
        optimizer, scheduler = self.get_optimizer(num_training_steps=t_total)
        # train
        total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )

        logger.info("***** Running training *****")
        logger.info("  Num datas = %d",len(train_dataset))
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss = 0.0
        logging_loss = 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())
            for step, input_tuples in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                tr_loss += self._training_step(input_tuples, optimizer)
                # 下面这个判断条件基本为true
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss
                        self._log(logs)
                        if self.args.evaluate_during_training:
                            self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(self.model, "module"):
                            assert self.model.module is self.model
                        else:
                            assert self.model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                        self.save_model(output_dir)
                        if self.is_world_master():
                            self._rotate_checkpoints()
                        if self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def evaluate(self):
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
