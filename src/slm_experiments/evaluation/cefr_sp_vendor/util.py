"""Vendored CEFR-SP utilities (trimmed for inference + checkpoint load)."""

from __future__ import annotations

import os

import numpy as np
import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.data_len = len(encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return self.data_len


class CEFRDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, sent_levels_a, sent_levels_b):
        self.encodings = encodings
        self.slabels_low = np.minimum(sent_levels_a, sent_levels_b)
        self.slabels_high = np.maximum(sent_levels_a, sent_levels_b)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["slabels_low"] = self.slabels_low[idx].clone().detach()
        item["slabels_high"] = self.slabels_high[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.slabels_high)


def read_corpus(path, num_labels):
    del num_labels  # upstream signature; labels always 6-way after convert
    levels_a, levels_b, sents = [], [], []
    with open(path) as f:
        for line in f:
            array = line.strip().split("\t")
            sents.append(array[0].split(" "))
            levels_a.append(float(array[1]) - 1)
            levels_b.append(float(array[2]) - 1)

    return np.array(levels_a), np.array(levels_b), sents


def convert_numeral_to_six_levels(levels):
    level_thresholds = np.array([0.0, 0.5, 1.5, 2.5, 3.5, 4.5])
    return _conversion(level_thresholds, levels)


def _conversion(level_thresholds, values):
    thresh_array = np.tile(level_thresholds, reps=(values.shape[0], 1))
    array = np.tile(values, reps=(1, level_thresholds.shape[0]))
    levels = np.maximum(
        np.zeros((values.shape[0], 1)),
        np.count_nonzero(thresh_array <= array, axis=1, keepdims=True) - 1,
    ).astype(int)
    return levels


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def token_embeddings_filtering_padding(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return token_embeddings * input_mask_expanded


def corpus_train_path_exists(corpus_path: str) -> bool:
    return bool(corpus_path) and os.path.isfile(f"{corpus_path}_train.txt")
