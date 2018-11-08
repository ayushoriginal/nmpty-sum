# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from . import FF

from ..utils.data import sort_batch


class VideoEncoder(nn.Module):
    """A recurrent encoder with embedding layer.

    Arguments:
        input_size (int): Embedding dimensionality.
        hidden_size (int): RNN hidden state dimensionality.
        n_vocab (int): Number of tokens for the embedding layer.
        rnn_type (str): RNN Type, i.e. GRU or LSTM.
        num_layers (int, optional): Number of stacked RNNs (Default: 1).
        bidirectional (bool, optional): If `False`, the RNN is unidirectional.
        dropout_rnn (float, optional): Inter-layer dropout rate only
            applicable if `num_layers > 1`. (Default: 0.)
        dropout_emb(float, optional): Dropout rate for embeddings (Default: 0.)
        dropout_ctx(float, optional): Dropout rate for the
            encodings/annotations (Default: 0.)
        emb_maxnorm(float, optional): If given, renormalizes embeddings so
            that their norm is the given value.
        emb_gradscale(bool, optional): If `True`, scales the gradients
            per embedding w.r.t. to its frequency in the batch.

    Input:
        x (Variable): A variable of shape (n_timesteps, n_samples)
            including the integer token indices for the given batch.
    Output:
        hs (Variable): A variable of shape (n_timesteps, n_samples, hidden)
            that contains encoder hidden states for all timesteps. If
            bidirectional, `hs` is doubled in size in the last dimension
            to contain both directional states.
        mask (Variable): A binary mask of shape (n_timesteps, n_samples)
            that may further be used in attention and/or decoder.
    """
    def __init__(self, input_size, proj_size, hidden_size, rnn_type,
                 num_layers=1, bidirectional=True,
                 dropout_rnn=0, dropout_emb=0, dropout_ctx=0,
                 emb_maxnorm=None, emb_gradscale=False):
        super().__init__()

        self.rnn_type = rnn_type.upper()
        self.input_size = input_size
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale

        # For dropout btw layers, only effective if num_layers > 1
        self.dropout_rnn = dropout_rnn

        # Our other custom dropouts after embeddings and annotations
        self.dropout_emb = dropout_emb
        self.dropout_ctx = dropout_ctx

        self.ctx_size = self.hidden_size
        # Doubles its size because of concatenation
        if self.bidirectional:
            self.ctx_size *= 2

        if self.dropout_emb > 0:
            self.do_emb = nn.Dropout(self.dropout_emb)
        if self.dropout_ctx > 0:
            self.do_ctx = nn.Dropout(self.dropout_ctx)

        # Create a video frame embedding layer that maps 2048 -> proj_size
        # This is an atypical embedding: add a bias + non-linear activation
        self.emb = FF(self.input_size, self.proj_size, bias=True, activ='tanh')

        # Create fused/cudnn encoder according to the requested type
        RNN = getattr(nn, self.rnn_type)
        self.enc = RNN(self.proj_size, self.hidden_size,
                       self.num_layers, bias=True, batch_first=False,
                       dropout=self.dropout_rnn,
                       bidirectional=self.bidirectional)

    def forward(self, x):
        '''
        We assume the batch has equally timestepped data that comes out of the bucket.
        '''
        # Embed the video feature vectors using a Linear layer
        embs = self.emb(x.float())

        # Transpose embs because the RNN demands a different order
        # (n_samples, timesteps, feats) -> (timesteps, n_samples, features)
        embs = embs.transpose(0,1)

        if self.dropout_emb > 0:
            embs = self.do_emb(embs)

        # Encode
        hs, _ = self.enc(embs)

        if self.dropout_ctx > 0:
            hs = self.do_ctx(hs)

        # No mask is returned as batch contains elements of all same lengths
        return hs, None
