import csv
import copy
import pickle
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable, Optional
import sys
sys.path.append("..")
from log import get_logger
logger = get_logger('root')

class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self, guid, text_a, text_b=None, label=None, meta: Optional[Dict] = None, idx=-1):
        """
        Create a new InputExample.
        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.idx = idx
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class InputFeatures(object):
    """A set of numeric features obtained from an :class:`InputExample`"""

    def __init__(self, input_ids, attention_mask, token_type_ids, label, meta: Optional[Dict] = None, idx=-1):
        """
        Create new InputFeatures.
        :param input_ids: the input ids corresponding to the original text or text sequence
        :param token_type_ids: segment ids as used by BERT
        :param label: the label
        :param mlm_labels: an optional sequence of labels used for auxiliary language modeling
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index

        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.idx = idx
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def pretty_print(self, tokenizer):
        return f'input_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n' + \
               f'attention_mask    = {self.attention_mask}\n' + \
               f'token_type_ids    = {self.token_type_ids}\n' + \
               f'label             = {self.label}\n'  + \
               f'idx               = {self.idx}\n'

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples

class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading train/dev32/dev/test/unlabeled examples for a given task.
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_dev32_examples(self, data_dir) -> List[InputExample]:
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass

class BoolQProcessor(DataProcessor):
    """Processor for the BoolQ data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["False", "True"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = str(example_json['label']) if 'label' in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['passage']
                text_b = example_json['question']
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)

        return examples


class MultiRcProcessor(DataProcessor):
    """Processor for the MultiRC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)

                passage_idx = example_json['idx']
                text = example_json['passage']['text']
                questions = example_json['passage']['questions']
                for question_json in questions:
                    question = question_json["question"]
                    question_idx = question_json['idx']
                    answers = question_json["answers"]
                    for answer_json in answers:
                        label = str(answer_json["label"]) if 'label' in answer_json else None
                        answer_idx = answer_json["idx"]
                        guid = f'{set_type}-p{passage_idx}-q{question_idx}-a{answer_idx}'
                        meta = {
                            'passage_idx': passage_idx,
                            'question_idx': question_idx,
                            'answer_idx': answer_idx,
                            'answer': answer_json["text"]
                        }
                        idx = [passage_idx, question_idx, answer_idx]
                        example = InputExample(guid=guid, text_a=text, text_b=question, label=label, meta=meta, idx=idx)
                        examples.append(example)

        # question_indices = list(set(example.meta['question_idx'] for example in examples))
        # label_distribution = Counter(example.label for example in examples)
        # logger.info(f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
        #             f"distribution {list(label_distribution.items())}")
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the Mnli data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.tsv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev_matched.tsv"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test_matched.csv"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for lineno,line in enumerate(f):
                if lineno==0:
                    continue
                line=line.strip().split('\t')
                idx = line[0]
                guid = f'{set_type}-{idx}'
                label = line[-1]
                text_a = line[8]
                text_b = line[9]
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)
        return examples

class SnliProcessor(DataProcessor):
    """Processor for the Mnli data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "snli_1.0_train.txt"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "snli_1.0_dev.txt"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "snli_1.0_test.txt"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for lineno,line in enumerate(f):
                if lineno==0:
                    continue
                line=line.strip().split('\t')
                idx = lineno
                guid = f'{set_type}-{idx}'
                label = line[0]
                if label == '-':
                    continue
                text_a = line[5]
                text_b = line[6]
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)
        return examples

class RteProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.tsv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.tsv"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.tsv"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for lineno,line in enumerate(f):
                if lineno==0:
                    continue
                line = line.strip().split('\t')
                idx = line[0]
                guid = "%s-%s" % (set_type, idx)
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)
        return examples    


# type: Dict[str,Callable[[],DataProcessor]]
PROCESSORS = {
    "boolq": BoolQProcessor,
    "multirc": MultiRcProcessor,
    'mnli': MnliProcessor,
    'rte': RteProcessor,
    "snli": SnliProcessor
} 

TASKS_MAPPING={
    "cola": "CoLA",
    "mnli": "MNLI",
    "mrpc": "MRPC",
    "qnli": "QNLI",
    "qqp":  "QQP",
    "rte": "RTE",
    "sst2": "SST-2",
    "stsb": "STS-B",
    "wnli": "WNLI",
    "snli": "SNLI"
}


TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"
DEV32_SET = "dev32"


SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET, DEV32_SET]

def load_examples(task, data_dir: str, set_type: str, *_, num_examples: int = None,
                  num_examples_per_label: int = None, seed: int = 42) -> List[InputExample]:
    """Load examples for a given task."""

    assert (not set_type == UNLABELED_SET) or (num_examples is not None), \
        "For unlabeled data, 'num_examples_per_label' is not allowed"

    processor = PROCESSORS[task]()

    ex_str = f"num_examples={num_examples}" if num_examples is not None \
        else f"num_examples_per_label={num_examples_per_label}"
    logger.info(
        f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})"
    )

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == DEV32_SET: ### TODO
        examples = processor.get_dev32_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    elif set_type == UNLABELED_SET:
        examples = processor.get_unlabeled_examples(data_dir)
        for example in examples:
            example.label = processor.get_labels()[0]
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")

    if num_examples is not None:
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(processor.get_labels(), num_examples_per_label)
        for example in examples:
            limited_examples.add(example)
        examples = limited_examples.to_list()

    label_distribution = Counter(example.label for example in examples)
    logger.info(f"Task: {task}, Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")

    return examples



if __name__=='__main__':
    dataset_name='mnli'
    data_dir="/home/yanzhongxiang/XPROMPT/datasets/MNLI"
    train_data=load_examples(dataset_name,data_dir,"train")
