# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from pathlib import Path

import shelve
import functools

from sklearn import preprocessing
import numpy as np
from torch.utils.data import Dataset
from ..utils.data import pad_video_sequence


class ShelveDataset(Dataset):
    r"""A PyTorch dataset for Shelve serialized tensor files. The
    serialized tensor's first dimension should be the batch dimension.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object for
            the relevant .shelve file.
        norm_and_scale: True or False: Should we normalise and scale
            the image features?
    """

    def __init__(self, fname, key=None, norm_and_scale=False):
        self.path = Path('{}.dat'.format(fname, 'dat'))
        if not self.path.exists():
            raise RuntimeError('{} does not exist.'.format(self.path))

        self.data = shelve.open(str(fname.resolve()))
        self.norm_and_scale = norm_and_scale

        # Dataset size
        self.size = len(self.data)

        # Stores the lengths of the input video sequences to enable bucketing
        self.lengths = self.read_sequence_lengths()

    def read_sequence_lengths(self):
        '''Returns an array with the number of video feature vectors
        stored for each image. TODO: This is expensive and a slow
        way to start the process.'''
        lengths = []
        for x in self.data:
            lengths.append(len(self.data[str(x)]))
        return lengths

    @staticmethod
    def to_torch(batch):
        return pad_video_sequence(batch)

    # Efficiency hack: We have set an unbounded lru_cache.
    # This will eat a lot of RAM at the cost of less Overhead time
    # per epoch when training
    @functools.lru_cache(maxsize=5000)
    def __getitem__(self, idx):
        if self.norm_and_scale:
            feats = self.data[str(idx)]
            feats = preprocessing.normalize(feats)
            return feats
        else:
            return np.array(self.data[str(idx)])

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} samples)\n".format(
            self.__class__.__name__, self.path.name, self.__len__())
        return s
