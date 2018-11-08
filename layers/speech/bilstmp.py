# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.nn import functional as F

from ..ff import FF


class BiLSTMp(nn.Module):
    """A bidirectional LSTM encoder for speech features. A batch should
    only contain samples that have the same sequence length.

    Arguments:
        input_size (int): Input feature dimensionality.
        hidden_size (int): LSTM hidden state dimensionality.
        layers (tuple(int)): A tuple giving the subsampling factor for each layer.
        activ (str, optional): Non-linearity to apply to intermediate projection
            layers. (Default: 'tanh')
        dropout (float, optional): Use dropout (Default: 0.)
    Input:
        x (Variable): A variable of shape (n_timesteps, n_samples, n_feats)
            that includes acoustic features of dimension ``n_feats`` per
            each timestep (in the first dimension).

    Output:
        hs (Variable): A variable of shape (n_timesteps, n_samples, hidden * 2)
            that contains encoder hidden states for all timesteps.
        mask (Variable): A binary mask of shape (n_timesteps, n_samples)
            that may further be used in attention and/or decoder. `None`
            is returned for now as homogeneous batches are expected.
    """
    def __init__(self, input_size, hidden_size, layers, activ='tanh', dropout=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.n_layers = len(self.layers)
        self.dropout = dropout
        self.activ = activ

        # Doubles its size because of concatenation of forw-backw encs
        self.ctx_size = self.hidden_size * 2

        # Fill 0-vector as <eos> to the end of the frames
        self.pad_tuple = (0, 0, 0, 0, 0, 1)

        self.ffs = nn.ModuleList()
        self.lstms = nn.ModuleList()

        for i, ss_factor in enumerate(self.layers):
            # Add LSTMs
            self.lstms.append(nn.LSTM(
                self.input_size if i == 0 else self.hidden_size,
                self.hidden_size, dropout=self.dropout,
                bidirectional=True))
            # Add non-linear bottlenecks
            self.ffs.append(FF(
                self.ctx_size, self.hidden_size, activ=self.activ))

    def forward(self, x):
        # Pad with <eos> zero
        hs = F.pad(x, self.pad_tuple)

        for ss_factor, (f_lstm, f_ff) in enumerate(zip(self.lstms, self.ffs)):
            if ss_factor > 1:
                # Skip states
                hs = f_ff(f_lstm(hs[::ss_factor])[0])
            else:
                hs = f_ff(f_lstm(hs)[0])

        # No mask is returned as batch contains elements of all same lengths
        return hs, None
