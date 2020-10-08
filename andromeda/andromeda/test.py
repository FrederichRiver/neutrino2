#!/usr/bin/python38
import torch
import torch.nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.data.dataset import Dataset
from transformers import get_linear_schedule_with_warmup
from andromeda.model_config import bert_config, train_config
from pathlib import Path
from andromeda.model import CNerTokenizer, BertCRFModel
from andromeda.model_tool import (
    CNERProcessor, InputFeatures, total_train_step, set_seed)
from andromeda.model_tool import collate_fn
import logging
import os


logger = logging.getLogger(__name__)


def convert_examples_to_features(
            examples: list, label_list: list, tokenizer, max_seq_length=768, cls_token="[CLS]", cls_token_segment_id=1, sep_token="[SEP]", pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True):
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


def load_and_cache(args: dict, tokenizer) -> TensorDataset:
    """
    For CNER dataset.\n
    Return: (input_ids, attention_mask, segment_ids, labels_ids, lens)
    """
    # Load data features from cache or dataset file
    feature_param = {
        "max_seq_length": 512,
        "cls_token": "[CLS]",
        "cls_token_segment_id": 1,
        "sep_token": "[SEP]",
        "pad_token": 0,
        "pad_token_segment_id": 0,
        "sequence_a_segment_id": 0,
        "mask_padding_with_zero": True
    }
    processor = CNERProcessor()
    label_list = processor.get_labels()
    examples = processor.get_train_examples(args['data_dir'])
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            **feature_param
                                            )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def train(args: dict, model, train_dataloader: DataLoader, train_dataset: Dataset, tokenizer):
    """ Train the model """
    # train_sampler = SequentialSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate_fn)
    train_step = total_train_step(args['max_steps'], len(train_dataloader), args['gradient_accumulation_steps'], args['num_train_epochs'])
    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = optim_config(train_config, model)
    scheduler = scheduler_config(train_config, optimizer, train_step)
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
    tr_loss = 0.0
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
            # print(inputs)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            model.zero_grad()
            if args['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            else:
                loss.backward()
            tr_loss += loss.item()
            # if (step + 1) % args['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            if (step + 1) % 20 == 0:
                save_model(train_config, model, optimizer, scheduler, tokenizer)
    return global_step, tr_loss / global_step


def save_model(args, model, optimizer, scheduler, tokenizer):
    if model:
        torch.save(model.state_dict(), os.path.join(args["output_dir"], "model.pkl"))
    if optimizer:
        torch.save(optimizer.state_dict(), os.path.join(args["output_dir"], "optimizer.pt"))
    if scheduler:
        torch.save(scheduler.state_dict(), os.path.join(args["output_dir"], "scheduler.pt"))
    if tokenizer:
        tokenizer.save_vocabulary(args["output_dir"])


def optim_config(args: dict, model):
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    if os.path.isfile(os.path.join(args['model_name_or_path'], "optimizer.pt")):
        # Load in optimizer states
        optimizer.load_state_dict(torch.load(os.path.join(args['model_name_or_path'], "optimizer.pt")))
    return optimizer


def scheduler_config(args: dict, optimizer, train_step):
    args['warmup_steps'] = int(train_step * args['warmup_proportion'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'],
                                                num_training_steps=train_step)
    if os.path.isfile(os.path.join(args['model_name_or_path'], "scheduler.pt")):
        # Load in optimizer and scheduler states
        scheduler.load_state_dict(torch.load(os.path.join(args['model_name_or_path'], "scheduler.pt")))
    return scheduler


if __name__ == "__main__":
    import time
    # if not os.path.exists(args['output_dir']):
    #    os.mkdir(args['output_dir'])
    # args['output_dir'] = args['output_dir'] + '{}'.format(args.model_type)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # init_logger(log_file=train_config["output_dir"] + f'/model-type-task-name-{time_}.log')
    processor = CNERProcessor()
    label_list = processor.get_labels()
    var_id2label = {i: label for i, label in enumerate(label_list)}
    var_label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    # 'bert': (BertConfig, BertCrfForNer, CNerTokenizer)
    tokenizer = CNerTokenizer.from_pretrained('bert-base-chinese', config=bert_config, cache_dir=train_config['cache_dir'], bos_token='[BOS]', eos_token='[EOS]')
    bert_model = BertCRFModel.from_pretrained('bert-base-chinese', config=bert_config)
    train_dataset = load_and_cache(train_config, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=10)
    global_step, tr_loss = train(train_config, bert_model, train_dataloader, train_dataset, tokenizer)
