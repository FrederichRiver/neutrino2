#!/usr/bin/python38
import csv
import copy
import json
import os
import random
import torch
import torch.cuda
import torch.backends.cudnn
import numpy as np


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """
        Gets the list of labels for this data set.\n
        Return a list of string labels like ['B', ..., 'X', 'O']
        """
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file: str, quotechar=None) -> list:
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(cls, input_file: str) -> list:
        """
        Text file should start with string '-DOCSTART-' or
        section with empty line or enter symbol.
        """
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def _read_json(cls, input_file) -> list:
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index+1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-'+key
                                else:
                                    labels[start_index] = 'B-'+key
                                    labels[start_index+1:end_index+1] = ['I-'+key]*(len(sub_name)-1)
                lines.append({"words": words, "labels": labels})
        return lines


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid: str, text_a: list, labels: list):
        """
        Args:
        >guid: Unique id for the example.\n
        >text_a: list. The words of the sequence.\n
        >labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids: list, input_mask: list, input_len: int, segment_ids: list, label_ids: list):
        """
        Args:\n
        >input_ids: list, a list of tokens from tokenizer.\n
        >input_mask: list of int, mask equals to one if the token is from sentence,
                        else equals zero means the token is a padding token.\n
        >segment_ids: \n
        >label_ids: \n
        >input_len: int, it is the real size of input_ids without padding tokens. \n
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
            examples: InputExample, label_list: list, tokenizer, max_seq_length: int, cls_token="[CLS]", cls_token_segment_id=1, sep_token="[SEP]", pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True):
    """
    Loads a data file into a list of `InputBatch`s\n
    The patten of token is like [CLS] + A + [SEP] + B + [SEP] for BERT.\n
    cls_token_segment_id: define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # Convert label_list into a dict like {label: i}
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    # Account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    for (ex_index, example) in enumerate(examples):
        # convert example into features list.
        # if ex_index % 10000 == 0:
        #    logger.info("Writing example %d of %d", ex_index, len(examples))
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels]
        # Cut the tokens if tokens is longer than max_seq_length.
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [label_map['O']]  # Here O is the letter O.
        segment_ids = [sequence_a_segment_id] * len(tokens)
        # Add [CLS] to the head of token list.
        tokens = [cls_token] + tokens
        label_ids = [label_map['O']] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids
        # Code for adding [CLS] token ENDs here.
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        # Pad on right.
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token] * padding_length
        """
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        """
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len, segment_ids=segment_ids, label_ids=label_ids))
    return features  # features is a list of InputFeatures


def collate_fn(batch: tuple):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    # max_len = max(all_lens).item()
    max_len = 768
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    #      batch[0]      batch[1]             batch[2]           batch[3]     batch[4]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


def total_train_step(max_step: int, len_train_data: int, gradient_accumulation_step: int, num_train_epochs: int):
    if max_step > 0:
        total_step = max_step
        num_train_epochs = max_step // (len_train_data // gradient_accumulation_step) + 1
    else:
        total_step = len_train_data // gradient_accumulation_step * num_train_epochs
    return total_step


def set_seed(seed=1029):
    '''
    Setting seed for entire development environment.
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True