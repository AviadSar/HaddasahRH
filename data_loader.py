import torch
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from datasets import load_dataset
import argparse
from args_classes import DataLoaderArgs

from processing_funcs import get_processing_func_from_args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_file",
        help="path to a json file to load arguments from",
        type=str,
        default=None
    )

    parser.add_argument(
        "--data_file",
        help="path to the newly created dataset directory",
        type=str,
        default="C:\\Users\\aavia\\PycharmProjects\\HaddasahRH\\data\\social_assesments_100_annotations_en.tsv"
    )

    parser.add_argument(
        "--processed_data_dir",
        help="path to the newly created dataset directory",
        type=str,
        default="C:\\my_documents\\datasets\\AMNLPFinal\\miss_last_sentence_5_sentences_"
    )

    parser.add_argument(
        '--n_train_samples',
        help='continue or start training from this epoch',
        type=int,
        default=50,
    )

    parser.add_argument(
        '--n_dev_samples',
        help='continue or start training from this epoch',
        type=int,
        default=50,
    )

    parser.add_argument(
        '--processing_func',
        help='the function used to adjust the newly created dataset',
        type=str,
        default=None
    )

    args = parser.parse_args()
    if args.json_file:
        args = DataLoaderArgs(args.json_file)
    return args


def write_data_as_csv(data, data_dir):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        print("creating dataset directory " + data_dir)

    train, dev, test = data
    train.to_csv(data_dir + os.path.sep + 'train.tsv', sep='\t')
    dev.to_csv(data_dir + os.path.sep + 'dev.tsv', sep='\t')
    test.to_csv(data_dir + os.path.sep + 'test.tsv', sep='\t')


def read_data_from_csv(data_dir):
    train = pd.read_csv(data_dir + os.path.sep + 'train.tsv', sep='\t')
    dev = pd.read_csv(data_dir + os.path.sep + 'dev.tsv', sep='\t')
    test = pd.read_csv(data_dir + os.path.sep + 'test.tsv', sep='\t')

    return train, dev, test


def load_data(args):
    data = pd.read_csv(args.data_file, sep='\t')
    n_train_samples = args.n_train_samples
    n_dev_samples = args.n_dev_samples

    data = args.processing_func(data)

    train = data[:n_train_samples]
    dev = data[n_train_samples: n_train_samples + n_dev_samples]
    data = data[:n_train_samples].append(data[n_train_samples + n_dev_samples:])

    return [train, dev, data]


if __name__ == '__main__':
    args = parse_args()
    args.processing_func = get_processing_func_from_args(args)

    np.random.seed(42)
    load_data(args)
