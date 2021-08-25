import torch
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from datasets import load_dataset
import argparse
from args_classes import Args

from patterns import get_patterns_from_args


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


def load_data(args):
    data = pd.read_csv(args.data_file, sep='\t')
    n_train_samples = args.n_train_samples
    n_dev_samples = args.n_dev_samples
    n_test_samples = args.n_test_samples

    patterns = get_patterns_from_args(args)
    args.patterns = patterns

    train = data[:n_train_samples]
    dev = data[n_train_samples: n_train_samples + n_dev_samples]
    test = data[n_train_samples + n_dev_samples: n_train_samples + n_dev_samples + n_test_samples]
    data = data[:n_train_samples + n_dev_samples].append(data[n_train_samples + n_dev_samples + n_test_samples:])

    return train, dev, test, data
