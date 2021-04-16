#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/21 9:29 上午
# @Author  : SPC
# @FileName: SemProcessor.py
# @Software: PyCharm
# @Desn    ：
import logging
import json
import os
import dataclasses
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Union
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
logger = logging.getLogger(__name__)

@dataclass
class InputExamplePairwise:
    guid: Optional[str] = None
    qid: Optional[int] = None
    text_a: Optional[str] = None
    text_b: Optional[str] = None
    label: Optional[str] = None
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

def glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    return _glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )

def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features

def pairwise_convert_examples_to_features(
    examples: Union[List[InputExamplePairwise], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    return _pairwise_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )

def _pairwise_convert_examples_to_features(
    examples: List[InputExamplePairwise],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExamplePairwise) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append((int(examples[i].qid),feature,examples[i].label))

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i][1])

    return features

class SemSimProcessor(DataProcessor):
    """Processor for the Text Classification data set"""
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )
    def _read_data(self,path):
        fs = open(path,"r",encoding="UTF-8")
        lines = fs.readlines()
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            ls = line.strip().split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = ls[0]
            text_b = ls[1]
            label = None if set_type == "test" else ls[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class PairwiseSemSimProcessor(DataProcessor):

    def _read_data(self,path):
        fs = open(path,"r",encoding="UTF-8")
        lines = fs.readlines()
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            ls = line.strip().split("\t")
            guid = "%s-%s" % (set_type, i)
            qid = ls[0]
            text_a = ls[1]
            text_b = ls[2]
            label = None if set_type == "test" else ls[3]
            examples.append(InputExamplePairwise(guid=guid, qid=qid, text_a=text_a, text_b=text_b, label=label))
        return examples

class OutputMode(Enum):
    classification = "classification"
    regression = "regression"

tasks_num_labels = {
    "sim": 2,
}

processors = {
    "sim": SemSimProcessor,
}

output_modes = {
    "sim": "classification",
}