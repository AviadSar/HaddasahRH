import torch
import numpy as np
import os
import sklearn
import pandas as pd
import argparse

import patterns
from args_classes import Args
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaForSequenceClassification, Trainer, TrainingArguments,\
    TrainerCallback, RobertaConfig
from tokenizers import AddedToken
import data_loader
import dataset_classes
from datasets import load_metric
from logger import Logger, log_from_log_history
from data_loader import load_data

accuracy_metric = load_metric("accuracy")


class EvaluateAndSaveCallback(TrainerCallback):
    def on_step_end(self, callback_args, state, control, logs=None, **kwargs):
        if state.global_step % callback_args.eval_steps == 0:
            control.should_evaluate = True
            control.should_save = True


class LoggingCallback(TrainerCallback):
    def on_step_end(self, callback_args, state, control, logs=None, **kwargs):
        if state.global_step % callback_args.logging_steps == 0:
            control.should_log = True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_file",
        help="path to a json file to load arguments from",
        type=str,
        default=None
    )

    args = parser.parse_args()
    if args.json_file:
        args = Args(args.json_file)
    return args


def get_model_and_tokenizer(args, type):
    model, tokenizer = None, None
    if 'roberta' in args.model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
        if type == 'sequence_classification':
            model = RobertaForSequenceClassification.from_pretrained(args.model_name,
                                                                     hidden_dropout_prob=args.dropout,
                                                                     attention_probs_dropout_prob=args.dropout,
                                                                     num_labels=args.num_labels)
        elif type == 'masked_LM':
            model = RobertaForMaskedLM.from_pretrained(args.model_name,
                                                                  hidden_dropout_prob=args.dropout,
                                                                  attention_probs_dropout_prob=args.dropout)
    if args.eval:
        model = model.from_pretrained(args.model_dir)
    if model and tokenizer:
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer
    raise Exception('no such model: name "{}", type "{}"'.format(args.model_name, args.model_type))


def compute_token_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions[labels != -100], references=labels[labels != -100])


def compute_sequence_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def compute_metrics(args):
    if args.model_type == 'token_classification':
        return compute_token_accuracy
    elif args.model_type == 'sequence_classification':
        return compute_sequence_accuracy


def encode_targets_for_token_classification(batch_target, tokenizer):
    encoded_targets_list = tokenizer(batch_target, return_attention_mask=False, truncation=True, padding='max_length')['input_ids']
    encoded_targets = np.array(encoded_targets_list)
    encoded_targets[np.logical_and(encoded_targets != (len(tokenizer) - 2), encoded_targets != (len(tokenizer) - 1))] = -100
    encoded_targets[encoded_targets == (len(tokenizer) - 2)] = 0
    encoded_targets[encoded_targets == (len(tokenizer) - 1)] = 1

    return encoded_targets.tolist()


def encode_targets(batch_target, tokenizer, args):
    if args.model_type == 'sequence_classification':
        return batch_target
    elif args.model_type == 'token_classification':
        return encode_targets_for_token_classification(batch_target, tokenizer)


def tokenize_datasets(tokenizer, datasets, args):
    tokenized_datasets = []
    for dataset in datasets:
        text = dataset['text'].tolist()
        target = dataset['target'].tolist()

        tokenized_datasets.append(
            {
                'encoded_text': tokenizer(text, max_length=512, return_attention_mask=True,
                                          truncation=True, padding='max_length'),
                'encoded_target': encode_targets(target, tokenizer, args)
            }
        )

    datasets = []
    for dataset in tokenized_datasets:
        datasets.append(dataset_classes.TextDataset(dataset['encoded_text'], dataset['encoded_target']))

    return dataset


def set_trainer(model, tokenizer, train, eval, args):
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        max_steps=args.eval_steps * args.num_evals,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 4,
        gradient_accumulation_steps=128 // args.batch_size,
        eval_accumulation_steps=args.batch_size * 3,
        warmup_steps=500,
        weight_decay=0.01,
        save_strategy='no',
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_total_limit=1,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        callbacks=[EvaluateAndSaveCallback(), LoggingCallback()],
        compute_metrics=compute_metrics(args)
    )

    return trainer


def apply_pattern_to_all_datasets(pattern, verbalizer, datasets, args):
    new_datasets = []
    for dataset in datasets:
        new_datasets.append(patterns.apply_pattern(pattern, verbalizer, dataset, args))
    return new_datasets


def adjust_target_column(datasets, args):
    for dataset in datasets:
        for label_replacement in args.label_dictionary:
            dataset[args.target_column] = dataset[args.target_column].replace(label_replacement[0], label_replacement[1])


def soft_label_data(args):
    model, tokenizer = get_model_and_tokenizer(args, 'masked_LM')
    train, dev, test, data = load_data(args)
    adjust_target_column((train, dev, test, data), args)

    pattern_logits = []
    for pattern, verbalizer in zip(args.patterns, args.verbalizers):
        curr_train, curr_dev, curr_test, curr_data = apply_pattern_to_all_datasets(pattern, verbalizer, (train, dev, test, data), args)
        tokenized_train, tokenized_dev, tokenized_test, tokenized_data = tokenize_datasets(tokenizer, (curr_train, curr_dev, curr_test, curr_data), args)
        trainer = set_trainer(model, tokenizer, tokenized_train, tokenized_dev, args)
        trainer.train()

        trainer.eval_dataset = tokenized_data
        evaluation = trainer.evaluate()

        pattern_logits.append(get_pattern_logits())

    logits = np.mean(np.array(pattern_logits), axis=0)
    classifier_data = add_logits_targets(tokenized_data)

    return classifier_data, tokenized_test


if __name__ == "__main__":
    args = parse_args()
    train, test = soft_label_data(args)
    model, tokenizer = get_model_and_tokenizer(args, 'sequence_classification')
    trainer = set_trainer(model, tokenizer, train, test, args)

    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError as e:
        if 'No valid checkpoint' in e.args[0]:
            trainer.train()
        else:
            raise e

    trainer.save_model(args.model_dir)
    trainer.save_state()
    log_from_log_history(trainer.state.log_history, args.model_dir)
