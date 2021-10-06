import torch
import numpy as np
import os
import sklearn
import pandas as pd
import argparse
from scipy.special import softmax

from patterns import get_patterns_from_args
from verbalizers import get_verbalizers_from_args

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
from model_classes import RobertaForSoftLabelSequenceClassification, CompactRobertaForMaskedLM

accuracy_metric = load_metric("accuracy")


class EvaluateAndSaveCallback(TrainerCallback):
    def on_step_end(self, callback_args, state, control, logs=None, **kwargs):
        if state.global_step % callback_args.eval_steps == 0:
            control.should_evaluate = True
            control.should_save = True


class EvaluateCallback(TrainerCallback):
    def on_step_end(self, callback_args, state, control, logs=None, **kwargs):
        if state.global_step % callback_args.eval_steps == 0:
            control.should_evaluate = True


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


def get_model_and_tokenizer(args, type='pattern'):
    if type == 'pattern':
        dropout = args.pattern_dropout
    elif type == 'classifier':
        dropout = args.classifier_dropout
    else:
        raise ValueError('"type" argument for "get_model_and_tokenizer" mast be "pattern" or "classifier", not {}'.format(type))

    model, tokenizer = None, None
    if 'roberta' in args.model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
        if args.model_type == 'sequence_classification':
            model = RobertaForSequenceClassification.from_pretrained(args.model_name,
                                                                     hidden_dropout_prob=dropout,
                                                                     attention_probs_dropout_prob=dropout,
                                                                     num_labels=args.num_labels)
        elif args.model_type == 'MLM':
            model = CompactRobertaForMaskedLM.from_pretrained(args.model_name,
                                                                  hidden_dropout_prob=dropout,
                                                                  attention_probs_dropout_prob=dropout)
        elif args.model_type == 'soft_label_classification':
            model = RobertaForSoftLabelSequenceClassification.from_pretrained(args.model_name,
                                                                     hidden_dropout_prob=dropout,
                                                                     attention_probs_dropout_prob=dropout,
                                                                     num_labels=args.num_labels)
    if model and args.eval:
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


class compute_MLM_accuracy(object):
    def __init__(self, verbalizer, tokenizer, args):
        self.verbalizer = verbalizer
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, eval_pred):
        logits, labels = eval_pred

        class_token_idxs = [self.tokenizer.encode(self.verbalizer.classes[label])[1] for label in self.args.labels]
        for idx, class_token_idx in enumerate(class_token_idxs):
            labels[labels == class_token_idx] = idx

        predictions = np.argmax(logits, axis=-1)
        metric_dict = accuracy_metric.compute(predictions=predictions, references=labels[labels != -100])

        return metric_dict


class compute_MLM_logits(object):
    def __init__(self, verbalizer, tokenizer, args):
        self.verbalizer = verbalizer
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, eval_pred):
        logits, labels = eval_pred

        temperature = 2
        metric_dict = {'mask_probs': softmax(logits / temperature, axis=1)}

        return metric_dict


def compute_metrics(args):
    if args.model_type == 'token_classification':
        return compute_token_accuracy
    elif args.model_type == 'sequence_classification':
        return compute_sequence_accuracy


def encode_targets_for_token_classification(target, tokenizer):
    encoded_targets_list = tokenizer(target, return_attention_mask=False, truncation=True, padding='max_length')['input_ids']
    encoded_targets = np.array(encoded_targets_list)
    encoded_targets[np.logical_and(encoded_targets != (len(tokenizer) - 2), encoded_targets != (len(tokenizer) - 1))] = -100
    encoded_targets[encoded_targets == (len(tokenizer) - 2)] = 0
    encoded_targets[encoded_targets == (len(tokenizer) - 1)] = 1

    return encoded_targets.tolist()


def encode_targets_for_MLM(encoded_text, target, tokenizer):
    encoded_targets_list = tokenizer(target, return_attention_mask=False, truncation=True, padding='max_length')['input_ids']
    encoded_targets = np.array(encoded_targets_list)
    encoded_targets[np.array(encoded_text['input_ids']) != tokenizer.mask_token_id] = -100
    return encoded_targets.tolist()


def encode_targets(encoded_text, target, tokenizer, args):
    if args.model_type == 'sequence_classification':
        return target
    elif args.model_type == 'token_classification':
        return encode_targets_for_token_classification(target, tokenizer)
    elif args.model_type == 'MLM':
        return encode_targets_for_MLM(encoded_text, target, tokenizer)
    elif args.model_type == 'soft_label_classification':
        return target


def tokenize_datasets(tokenizer, datasets, args):
    tokenized_datasets = []
    for dataset in datasets:
        text = dataset['text'].tolist()
        target = dataset['target'].tolist()

        encoded_text = tokenizer(text, max_length=512, return_attention_mask=True, truncation=True, padding='max_length')
        encoded_target = encode_targets(encoded_text, target, tokenizer, args)
        tokenized_datasets.append({'encoded_text': encoded_text, 'encoded_target': encoded_target})

    datasets = []
    for dataset in tokenized_datasets:
        datasets.append(dataset_classes.TextDataset(dataset['encoded_text'], dataset['encoded_target']))

    return datasets


def set_trainer(model, train, eval, args, type='pattern'):
    if type == 'pattern':
        model_dir = args.pattern_model_dir
        batch_size = args.pattern_batch_size
        logging_steps = args.pattern_logging_steps
        eval_steps = args.pattern_eval_steps
        gradient_accumulation_steps = args.pattern_gradient_accumulation_steps
        num_evals = args.pattern_num_evals
        warmup_steps = args.pattern_warmup_steps
        callbacks = [EvaluateAndSaveCallback(), LoggingCallback()]
    elif type == 'classifier':
        model_dir = args.classifier_model_dir
        batch_size = args.classifier_batch_size
        logging_steps = args.classifier_logging_steps
        eval_steps = args.classifier_eval_steps
        gradient_accumulation_steps = args.classifier_gradient_accumulation_steps
        num_evals = args.classifier_num_evals
        warmup_steps = args.classifier_warmup_steps
        callbacks = [EvaluateAndSaveCallback(), LoggingCallback()]
    else:
        raise ValueError('trainer type mast be "pattern" or "classifier", not {}'.format(type))

    training_args = TrainingArguments(
        output_dir=model_dir,
        max_steps=eval_steps * num_evals,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # eval_accumulation_steps=8,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        save_strategy='no',
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_total_limit=1,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        callbacks=callbacks,
        compute_metrics=compute_metrics(args)
    )

    return trainer


def apply_pattern_to_all_datasets(pattern, verbalizer, datasets, input_only_datasets):
    for dataset in datasets:
        patterns.apply_pattern(pattern, verbalizer, dataset)
    for input_only_dataset in input_only_datasets:
        patterns.apply_pattern(pattern, verbalizer, input_only_dataset, input_only=True)


def adjust_help_at_home_hours(help_at_home_hours):
    if str(help_at_home_hours).replace('.', '', 1).isdigit():
        if float(help_at_home_hours) == 0:
            return 'no'
        elif float(help_at_home_hours) <= 20:
            return 'few'
        elif float(help_at_home_hours) < 100:
            return 'many'
        elif float(help_at_home_hours) == 100:
            return 'special'
    else:
        return help_at_home_hours


def adjust_children(children):
    if str(children).replace('.', '', 1).isdigit():
        if float(children) > 0:
            return 'yes'
        else:
            return 'no'
    else:
        return 'unknown'


def adjust_target_column(datasets, args):
    for dataset in datasets:
        for label_replacement in args.label_dictionary:
            dataset[args.target_column] = dataset[args.target_column].replace(label_replacement[0], label_replacement[1])
        if args.target_column == 'children':
            dataset[args.target_column] = dataset[args.target_column].apply(adjust_children)
        if args.target_column == 'help_at_home_hours':
            dataset[args.target_column] = dataset[args.target_column].apply(adjust_help_at_home_hours)


def get_pattern_probs(evaluation, verbalizer, tokenizer, args):
    # class_token_idxs = [tokenizer.encode(verbalizer.classes[label])[1] for label in args.labels]
    # pattern_probs = softmax(evaluation['eval_mask_logits'][:, class_token_idxs], axis=1)

    # pattern_logits_dict = {}
    # for target_class in verbalizer.classes.items():
    #     class_token_idx = tokenizer.encode(target_class[1])[1]
    #     pattern_logits_dict[target_class[0]] = evaluation['eval_mask_logits'][:, class_token_idx]
    # return pattern_logits_dict

    # return pattern_probs
    return evaluation['eval_mask_probs']


class labels_to_classes(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, label, *args, **kwargs):
        for idx, label_name in enumerate(self.args.labels):
            if label == label_name:
                return idx
        raise ValueError('label "{}" is not a viable label'.format(label))


def soft_label_data(args):
    args.model_type = 'MLM'
    model, tokenizer = get_model_and_tokenizer(args)
    train, dev, test, data = load_data(args)
    # data = data[:10]
    adjust_target_column((train, dev, test, data), args)

    pattern_probs = []
    pattern_accuracies = []
    for pattern, verbalizer in zip(args.patterns, args.verbalizers):
        model.verbalizer, model.tokenizer, model.args = verbalizer, tokenizer, args
        apply_pattern_to_all_datasets(pattern, verbalizer, [train, dev, test], [data])
        tokenized_train, tokenized_dev, tokenized_test, tokenized_data = tokenize_datasets(tokenizer, [train, dev, test, data], args)
        trainer = set_trainer(model, tokenized_train, tokenized_dev, args, type='pattern')
        trainer.compute_metrics = compute_MLM_accuracy(verbalizer, tokenizer, args)
        trainer.train()

        evaluation = trainer.evaluate()
        print("DEV")
        print(evaluation)
        pattern_accuracies.append(evaluation['eval_accuracy'])

        trainer.compute_metrics = compute_MLM_logits(verbalizer, tokenizer, args)
        trainer.eval_dataset = tokenized_data
        evaluation = trainer.evaluate()
        print(evaluation)
        pattern_probs.append(get_pattern_probs(evaluation, verbalizer, tokenizer, args))

    pattern_accuracies = np.array(pattern_accuracies) / sum(pattern_accuracies)
    data_target_probs = np.average(np.array(pattern_probs), weights=pattern_accuracies, axis=0)
    data['text'], test['text'] = data['social_assessment'], test['social_assessment']
    data['target'], test['target'] = list(data_target_probs), test[args.target_column].apply(labels_to_classes(args))

    train, dev = data[:(len(data) // 10) * 8], data[(len(data) // 10) * 8:]

    return train, dev, test


if __name__ == "__main__":
    args = parse_args()
    args.patterns = get_patterns_from_args(args)
    args.verbalizers = get_verbalizers_from_args(args)
    train, dev, test = soft_label_data(args)

    args.model_type = 'soft_label_classification'
    model, tokenizer = get_model_and_tokenizer(args)
    tokenized_train, tokenized_dev, tokenized_test = tokenize_datasets(tokenizer, (train, dev, test), args)
    trainer = set_trainer(model, tokenized_train, tokenized_dev, args, type='classifier')

    # try:
    #     trainer.train(resume_from_checkpoint=True)
    # except ValueError as e:
    #     if 'No valid checkpoint' in e.args[0]:
    #         trainer.train()
    #     else:
    #         raise e
    trainer.train()

    trainer.save_model(args.classifier_model_dir)
    trainer.save_state()

    args.model_type = 'sequence_classification'
    trainer.model.config.problem_type = None
    trainer.eval_dataset = tokenized_test
    trainer.compute_metrics = compute_sequence_accuracy
    evaluation = trainer.evaluate()
    print(evaluation)

    log_from_log_history(trainer.state.log_history, args.classifier_model_dir)
