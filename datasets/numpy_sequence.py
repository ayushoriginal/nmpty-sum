# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.misc import pbar


class NumpySequenceDataset(Dataset):
    """Read a sequence of numpy arrays.

    Arguments:
        fname (str or Path): Path to a list of paths to Numpy `.npy` files
            where each file contains an array with shape `(n_features, feat_dim)`.
            If the lines are in `<path>:<len>` format, additional length
            information will be used for bucketing. If the file itself is
            a `.npy` file, it will be treated as an array of numpy objects.
            For cases where all features are the same length, you should use
            `NumpyDataset`.
    """

    def __init__(self, fname):
        self.fname = fname
        self.data = []
        self.lengths = []
        self.has_lengths = False

        if not self.fname:
            raise RuntimeError('{} does not exist.'.format(self.fname))

        if str(self.fname).endswith('.npy'):
            self.data = np.load(self.fname)
            self.lengths = [x.shape[0] for x in self.data]
            self._read = lambda idx: self.data[idx]
            self.has_lengths = True
        else:
            self._read = lambda idx: np.load(self.data[idx])
            with open(self.fname) as f_list:
                # Detect file format and seek back
                self.has_lengths = ':' in f_list.readline()
                f_list.seek(0)
                for line in pbar(f_list, unit='sents'):
                    if self.has_lengths:
                        path, length = line.strip().split(':')
                        self.lengths.append(int(length))
                    else:
                        path = line.strip()
                    self.data.append(path)

        # Set dataset size
        self.size = len(self.data)

    @staticmethod
    def to_torch(batch):
        # List of (t, feat_dim)
        max_len = max(x.shape[0] for x in batch)
        width = batch[0].shape[1]
        padded = [np.zeros((max_len, width)) for _ in batch]
        for pad, x in zip(padded, batch):
            pad[:x.shape[0]] = x
        # padded is (n_samples, t, feat_dim)
        # return (n, f, t) for compatibility with the other input sources
        return torch.from_numpy(
            np.array(padded, dtype='float32')).transpose(1, 2)

    def __getitem__(self, idx):
        # Each item is (t, feat_dim)
        return self._read(idx)

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} (has_lengths={}) ({} samples)\n".format(
            self.__class__.__name__, self.has_lengths, self.__len__())
        s += " {}".format(self.fname)
        return s
