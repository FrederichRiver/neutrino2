#!/usr/bin/python38
import torch
import copy
import json
import torch.nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertForTokenClassification
from transformers import get_linear_schedule_with_warmup
from andromeda.model_config import bert_config, model_config_dict
from andromeda.crf import CRF
from pathlib import Path
from andromeda.model import CNerTokenizer, BertCRFModel
from andromeda.model_tool import (
    DataProcessor, InputExample, InputFeatures, total_train_step, set_seed)
from andromeda.model_tool import collate_fn
import logging
import os


logger = logging.getLogger(__name__)


def convert_examples_to_features(
            examples: InputExample, label_list: list, tokenizer, max_seq_length=768, cls_token="[CLS]", cls_token_segment_id=1, sep_token="[SEP]", pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True):
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
        # """
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        # """
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len, segment_ids=segment_ids, label_ids=label_ids))
    return features  # features is a list of InputFeatures


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0, 1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S


def load_and_cache(tokenizer):
    """
    Return: (input_ids, attention_mask, segment_ids, labels_ids, lens)
    """
    processor = CnerProcessor()
    # Load data features from cache or dataset file
    DATA_DIR = '/home/friederich/Documents/bert_model/data'
    cached_features_file = DATA_DIR + '/cached_features_file'
    feature_config = {
        "max_seq_length": 512,
        "cls_token": "[CLS]",
        "cls_token_segment_id": 1,
        "sep_token": "[SEP]",
        "pad_token": 0,
        "pad_token_segment_id": 0,
        "sequence_a_segment_id": 0,
        "mask_padding_with_zero": True
    }
    label_list = processor.get_labels()
    # if data_type == 'train':
    examples = processor.get_train_examples(DATA_DIR)
    # elif data_type == 'dev':
    #    examples = processor.get_dev_examples(DATA_DIR)
    # else:
    #    examples = processor.get_test_examples(DATA_DIR)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            **feature_config
                                            )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", 'B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                'O', 'S-NAME', 'S-ORG', 'S-RACE', "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #    continue
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                'B-organization', 'B-position', 'B-scene', "I-address",
                "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                'I-organization', 'I-position', 'I-scene',
                "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                'S-name', 'S-organization', 'S-position',
                'S-scene', 'O', "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


# Model config

# Prepare for data

# Training


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate_fn)
    train_step = total_train_step(args['max_steps'], len(train_dataloader), args['gradient_accumulation_steps'], args['num_train_epochs'])
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args['weight_decay'], 'lr': args['learning_rate']},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args['learning_rate']},
        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args['weight_decay'], 'lr': args['crf_learning_rate']},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args['crf_learning_rate']},
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args['weight_decay'], 'lr': args['crf_learning_rate']},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args['crf_learning_rate']}
    ]
    args['warmup_steps'] = int(train_step * args['warmup_proportion'])
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'],
                                                num_training_steps=train_step)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args['model_name_or_path'], "optimizer.pt")) and os.path.isfile(
            os.path.join(args['model_name_or_path'], "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args['model_name_or_path'], "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args['model_name_or_path'], "scheduler.pt")))
    # multi-gpu training (should be after apex fp16 initialization)
    # Train!
    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args['model_name_or_path']) and "checkpoint" in args['model_name_or_path']:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args['model_name_or_path'].split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args['gradient_accumulation_steps'])
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args['gradient_accumulation_steps'])
        print("  Continuing training from checkpoint, will skip to saved global_step")
        print("  Continuing training from epoch %d", epochs_trained)
        print("  Continuing training from global step %d", global_step)
        print("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    model.train()
    set_seed(args['seed'])  # Added here for reproductibility (even between python 2 and 3)
    for _ in range(int(args['num_train_epochs'])):
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # batch = tuple(t.to(args['device']) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
            inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            else:
                loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                """
                if args['local_rank'] in [-1, 0] and args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Log metrics
                    print(" ")
                    if args['local_rank'] == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, model, tokenizer)
                if args['local_rank'] in [-1, 0] and args['save_steps'] > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                """
                # os.system('exit')
        print(f"{global_step:=global_step}-{tr_loss:=tr_loss}")
        # logger.info("\n")
    return global_step, tr_loss / global_step


# Evaluation

# Report


# token = (input_ids, attention_mask, labels, input_lens)


"""class BertCRFModel(BertForTokenClassification):
    def __init__(self, config: BertConfig) -> None:
        super(BertCRFModel, self).__init__(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,)+outputs
        return outputs  # (loss), scores
"""

if __name__ == "__main__":
    import time
    # if not os.path.exists(args['output_dir']):
    #    os.mkdir(args['output_dir'])
    # args['output_dir'] = args['output_dir'] + '{}'.format(args.model_type)
    OUTPUT_DIR = '/home/friederich/Documents/bert_model/output'
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=OUTPUT_DIR + f'/model-type-task-name-{time_}.log')
    processor = CnerProcessor()
    label_list = processor.get_labels()
    var_id2label = {i: label for i, label in enumerate(label_list)}
    var_label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    # 'bert': (BertConfig, BertCrfForNer, CNerTokenizer)
    CACHE_DIR = '/home/friederich/Documents/bert_model/cache'
    print(bert_config)
    tokenizer = CNerTokenizer.from_pretrained('bert-base-chinese', config=bert_config, cache_dir=CACHE_DIR, bos_token='[BOS]', eos_token='[EOS]')
    bert_model = BertCRFModel.from_pretrained('bert-base-chinese', config=bert_config)
    TRAIN = True
    train_dataset = load_and_cache(tokenizer)
    global_step, tr_loss = train(model_config_dict, train_dataset, bert_model, tokenizer)
    print(" global_step = %s, average loss = %s", global_step, tr_loss)
    """
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    """
