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
import logging
from typing import Callable, Dict
from transformers import EvalPrediction
from sklearn.metrics import f1_score
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from new_model import BertWithTextCNN

def simple_accuracy(preds, labels):
    return (preds == labels).mean()
def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def main():
    parse = pbert_argparse.parse
    args = parse.parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Training Bert with Pairwise Strategy")
    try:
        num_labels = tasks_num_labels[args.task_name]
        output_mode = output_modes[args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (args.task_name))
    config = BertConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model = BertWithTextCNN.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    # Get datasets
    if args.do_train:
        if args.train_mode == "pointwise":
            train_dataset = SemSimDataset(args, tokenizer=tokenizer, mode="train")
        elif args.train_mode == "pairwise":
            train_dataset = PairwiseSemSimDataset(args, tokenizer=tokenizer, mode="train")
    else:
        train_dataset = None
    eval_dataset = (
        SemSimDataset(args, tokenizer=tokenizer, mode="eval") if args.do_eval else None
    )
    test_dataset = (
        SemSimDataset(args, tokenizer=tokenizer, mode="test") if args.do_predict else None
    )

    def build_compute_metrics_fn(task_name: str = None) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            return acc_and_f1(preds, p.label_ids)

        return compute_metrics_fn

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(args.task_name),
    )

    if args.do_train:
        trainer.train(
            model_path=args.model_name_or_path if os.path.isdir(args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(args.output_dir)

    # Evaluation
    eval_results = {}
    if args.do_eval:
        logger.info("*** Evaluate ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        logger.info("dev_length:" + str(len(eval_dataset)))
        if args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(args, task_name="mnli-mm")
            eval_datasets.append(
                SemSimDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=args.cache_dir)
            )

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
            eval_results.update(eval_result)

        if args.do_predict:
            logging.info("*** Test ***")
            test_datasets = [test_dataset]
            if args.task_name == "mnli":
                mnli_mm_data_args = dataclasses.replace(args, task_name="mnli-mm")
                test_datasets.append(
                    SemSimDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=args.cache_dir)
                )

            for test_dataset in test_datasets:
                predictions = trainer.predict(test_dataset=test_dataset).predictions
                if output_mode == "classification":
                    # predictions = np.argmax(predictions, axis=1)
                    predictions = predictions[:, 1]

                output_test_file = os.path.join(
                    args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
                )
                if trainer.is_world_master():
                    with open(output_test_file, "w") as writer:
                        logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(predictions):
                            if output_mode == "regression":
                                writer.write("%d\t%3.3f\n" % (index, item))
                            else:
                                logger.info("%d\t%3.3f\n" % (index, item))
                                # item = test_dataset.get_labels()[item]
                                writer.write("%d\t%s\n" % (index, item))
        return eval_results

if __name__ == '__main__':
    main()

