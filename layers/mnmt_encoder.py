# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class MNMTEncoder(nn.Module):
    """A recurrent encoder with embedding layer that can make
    use of auxiliary feature vectors.

    Arguments:
        emb_dim (int): Embedding dimensionality.
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
        feat_fusion(str, optional): Can be one of concat or prepend.
            (Default: concat)

    Input:
        x (Variable): A variable of shape (n_timesteps, n_samples)
            including the integer token indices for the given batch.
            The batch should contain equal-length sequences.
        feats (Variable): A variable of shape (n_samples, emb_dim)
            that contains a projection of the visual features.

    Output:
        hs (Variable): A variable of shape (n_timesteps, n_samples, hidden)
            that contains encoder hidden states for all timesteps. If
            bidirectional, `hs` is doubled in size in the last dimension
            to contain both directional states.
        mask (Variable): A binary mask of shape (n_timesteps, n_samples)
            that may further be used in attention and/or decoder. `None`
            is returned since we do not need mask in this encoder which
            assumes that batches contain equal-length sequences.
    """
    def __init__(self, emb_dim, hidden_size, n_vocab, rnn_type,
                 num_layers=1, bidirectional=True,
                 dropout_rnn=0, dropout_emb=0, dropout_ctx=0,
                 emb_maxnorm=None, emb_gradscale=False, feat_fusion='concat'):
        super().__init__()

        self.rnn_type = rnn_type.upper()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.n_vocab = n_vocab
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale
        self.feat_fusion = feat_fusion

        assert self.feat_fusion in ('concat', 'prepend'), \
            "feat_fusion should be concat or prepend"

        # When concatenated, the input size to the encoder becomes
        # the double of the embedding size
        self.input_size = self.emb_dim
        if self.feat_fusion == 'concat':
            self.input_size *= 2

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

        # Create embedding layer
        self.emb = nn.Embedding(self.n_vocab, self.emb_dim,
                                padding_idx=0, max_norm=self.emb_maxnorm,
                                scale_grad_by_freq=self.emb_gradscale)

        # Create fused/cudnn encoder according to the requested type
        RNN = getattr(nn, self.rnn_type)
        self.enc = RNN(self.input_size, self.hidden_size,
                       self.num_layers, bias=True, batch_first=False,
                       dropout=self.dropout_rnn,
                       bidirectional=self.bidirectional)

    def forward(self, x, feats):
        """Encodes the given input ``x`` by first converting the
        integers to embeddings and then processing them with an RNN.
        ``feats`` is combined with the embeddings in various ways
        depending on the ``feat_fusion`` argument of this class.

        Arguments:
            x (Variable): A variable of shape (n_timesteps, n_samples)
                including the integer token indices for the given batch.
                The batch should contain equal-length sequences.
            feats (Variable): A variable of shape (n_samples, emb_dim)
                that contains a projection of the visual features.
        """

        # Fetch embeddings corresponding to the token indices
        # embs.shape will be (n_timesteps, n_samples, emb_dim)
        embs = self.emb(x)

        if self.feat_fusion == 'concat':
            embs = torch.cat((embs, feats.expand_as(embs)), dim=-1)

        elif self.feat_fusion == 'prepend':
            embs = torch.cat((feats.unsqueeze(0), embs), dim=0)

        # Apply dropout
        if self.dropout_emb > 0:
            embs = self.do_emb(embs)

        # Encode with RNN
        hs, _ = self.enc(embs)

        # Apply another dropout over the encodings
        if self.dropout_ctx > 0:
            hs = self.do_ctx(hs)

        # Return the hidden states with a None mask
        return hs, None
