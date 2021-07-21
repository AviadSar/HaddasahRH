import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizerFast
from tokenizers import AddedToken
from data_loader import read_data_from_csv
from trainer import parse_args
import pandas as pd


def less_than_n_tokens(data, n):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken('<skip>', lstrip=True), AddedToken('<no_skip>', lstrip=True)]})

    # splits_ratio = [1, 1, 0]
    splits_ratio = [1]
    splits = []
    for split, ratio in zip([data], splits_ratio):
        text = split['social_assesment'].tolist()[:int(len(split['social_assesment']) * ratio)]
        n_samples = len(text)
        if n_samples == 0:
            continue

        batch_size = 10000
        batch_idx = 0
        while batch_idx * batch_size < n_samples:
            batch_text = text[batch_idx * batch_size: min((batch_idx + 1) * batch_size, n_samples)]

            encoded_texts = tokenizer(batch_text, return_attention_mask=False, truncation=False, padding=False)['input_ids']
            greater_than_n_indices = []
            for text_idx, encoded_text in enumerate(encoded_texts):
                text_length = len(encoded_text)
                if text_length > n:
                    greater_than_n_indices.append(text_idx)
            split = split.drop(greater_than_n_indices, axis=0)

            print('batch ' + str(batch_idx) + ' done.')
            batch_idx += 1

        splits.append(split)
        return splits


def data_tokens_histograms(data):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken('<skip>', lstrip=True), AddedToken('<no_skip>', lstrip=True)]})

    # splits_ratio = [1, 1, 0]
    splits_ratio = [1]
    for split, ratio in zip([data], splits_ratio):
        histogram = {}
        text = split['social_assesment'].tolist()[:int(len(split['social_assesment']) * ratio)]
        print('number of samples: ' + str(len(text)))
        n_samples = len(text)
        if n_samples == 0:
            continue

        batch_size = 10000
        batch_idx = 0
        while batch_idx * batch_size < n_samples:
            batch_text = text[batch_idx * batch_size: min((batch_idx + 1) * batch_size, n_samples)]

            encoded_texts = tokenizer(batch_text, return_attention_mask=False, truncation=False, padding=False)['input_ids']
            for encoded_text in encoded_texts:
                text_length = len(encoded_text)
                if text_length in histogram:
                    histogram[text_length] += 1
                else:
                    histogram[text_length] = 1
            print('batch ' + str(batch_idx) + ' done.')
            batch_idx += 1

        greater_than_512 = 0
        max_len_lines = max(histogram.keys()) + 1
        list_histogram = [0] * (max_len_lines + 1)
        for key, value in histogram.items():
            list_histogram[key] = value
            if key > 512:
                greater_than_512 += value

        print('maximal text length (in tokens): ' + str(max(histogram)))
        print('number of entries greater than 512: ' + str(greater_than_512))
        print('histogram values: ' + str(histogram))

        plt.hist(np.array(list_histogram), bins=max_len_lines)
        plt.show()


def data_histograms(data):
    for split in data:

        max_len_lines = 0
        for index, row in split.iterrows():
            len_lines = len(row['text'].split('\n\n'))
            if len_lines > max_len_lines:
                max_len_lines = len_lines

        histogram = [0] * (max_len_lines + 1)
        for index, row in split.iterrows():
            len_lines = len(row['text'].split('\n\n'))
            histogram[len_lines] += 1

        print(max(histogram))
        print(histogram)

        plt.hist(np.array(histogram), bins=max_len_lines)
        plt.show()


if __name__ == '__main__':
    args = parse_args()
    data = pd.read_csv("data/social_assesments_100_annotations_en.tsv", sep='\t')
    print('number of nan entries id: ' + str(len(data[data['social_assesment'].isna()])))
    data = data[~data['social_assesment'].isna()].reset_index()
    data = less_than_n_tokens(data, 500)[0]
    data_tokens_histograms(data)
