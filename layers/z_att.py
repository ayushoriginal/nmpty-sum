# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from . import FF, Attention
from ..utils.nn import ModuleDict

logger = logging.getLogger('nmtpytorch')


class ZSpaceAtt(nn.Module):
    """Latent "z" space for combining the results of multiple encoders
        in a multitask setup.
        This is done by attending to each input modalities and generate a
        fixed-length sequence of states.

    Arguments:
        ctx_size_dict (int dict): Dictionary with key-value pairs
            {encoder type : input size}
        z_size (int): Size of the z-space vectors
        z_len (int): Len of the z-space sequence
        z_type (str, optional): Design of network, i.e. single-layer, multi-
            layer, highway (Default: 'ff' i.e. single-layer feed-forward)
        dec_init (str): how to initialise the decoder [sum|mean] (Default: sum)
        att_type: attention type [mlp|dot] (Default: mlp)
        att_activ: activation function for attention [linear|sigmoid|tanh] (Default: tanh)
        att_bottleneck: attention mechanism hidden size ['ctx',int] (Default: 'ctx')
        att_temp (float): attention temperature (Default: 1.0)
        transform_ctx (Bool): should we transform ctx for att mechanism (Default: True)

    Input:
        x (Variable dict): Dictionary of encoder results with key-value pairs
            {modality : encoder_result}

    Output:
        z (Variable): A sequence of z_len-dimensional vectors of shape (z_size).

    TODO: allow for returning a sequence of z states (will require mask)
    """

    def __init__(self, ctx_size_dict, z_size, z_len=10, z_type=None,
                 dec_init='mean_ctx', att_type='mlp',
                 att_activ='tanh', att_bottleneck='ctx', att_temp=1.0,
                 transform_ctx=True, mlp_bias=False, dropout_out=0,
                 emb_maxnorm=None, emb_gradscale=False):
        super().__init__()

        self.ctx_size_dict = ctx_size_dict
        self.z_size = z_size
        self.z_len = z_len
        self.z_type = z_type.lower() if z_type else None

        # Other arguments
        self.att_type = att_type
        self.att_bottleneck = att_bottleneck
        self.att_activ = att_activ
        self.att_temp = att_temp
        self.transform_ctx = transform_ctx
        self.mlp_bias = mlp_bias
        self.dropout_out = dropout_out
        self.dec_init = dec_init

        assert self.dec_init in ('zero', 'mean_ctx'), "dec_init '{}' not known".format(dec_init)

        # Create an attention layer for each modality
        # FIXME: with the ff_dec_inits layers, it might be possible to manage that soon...
        s= set([size for size in ctx_size_dict.values()])
        assert(len(s) == 1), "Encoders vector sizes are not equal! This is not yet handled..."
        self.ctx_size = next(iter(s))

        # TODO: sharing weights between att. mechanisms is possible
        self.att = ModuleDict()
        for k in ctx_size_dict:
            self.att[k] = Attention(self.ctx_size_dict[k], self.z_size,
                             transform_ctx=self.transform_ctx,
                             mlp_bias=self.mlp_bias,
                             att_type=self.att_type,
                             att_activ=self.att_activ,
                             att_bottleneck=self.att_bottleneck,
                             temp=self.att_temp)

        # Create decoder layer necessary for attention
        self.dec = nn.GRUCell(self.ctx_size, self.z_size)

        # FIXME: several strategies to initialize the decoder can be considered
        # Set decoder initializer
        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init))

        # if init is not zero, then
        if self.dec_init != 'zero':
            self.ff_dec_init = FF(
                self.ctx_size,
                self.z_size, activ='tanh')

        # Fusion operation
        self.merge_op = self._merge_sum

        # Safety check
        assert self.z_type in (None, 'ff', 'multi', 'highway'), \
                "layer z_type '{}' not known".format(z_type)
        if not self.z_type:
            assert(len(set([size for size in ctx_size_dict.values()])) == 1), "Encoders vector sizes are not equal! Consider using z_type: FF in config."


    def _rnn_init_zero(self, ctx_dict):
        # NOTE: all ctx should have the same size at this point
        h_0 = torch.zeros(self.ctx_size, self.z_size) # * self.n_states) # <-- ?? was used in cond_decoder.py
        return Variable(h_0).cuda()

    def _rnn_init_mean_ctx(self, ctx_dict):
        # NOTE: averaging the mean of all modalities
        key = next(iter(ctx_dict))
        res = torch.autograd.Variable(torch.zeros(ctx_dict[key][0].shape[1:])).cuda()
        for e in ctx_dict.keys():
            ctx, ctx_mask = ctx_dict[e]
            if ctx_mask is None:
                res += ctx.mean(0)
            else:
                res += ctx.sum(0) / ctx_mask.sum(0).unsqueeze(1)
        return self.ff_dec_init(res / len(ctx_dict))

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        self.alphas = []
        return self._init_func(ctx_dict)

    def forward(self, ctx_dict):
        """
            ctx_dict is a dict of tuples (values, mask), simply operate on values
        """
        z_states_list = []

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # loop for z_len timesteps
        for t in range(self.z_len):
            # compute attended vector for each encoders
            self.txt_alpha_t = {}
            txt_z_t = {}
            for e in ctx_dict:
                # Apply attention
                self.txt_alpha_t[e], txt_z_t[e] = self.att[e](
                    h.unsqueeze(0), *ctx_dict[e])
            # merge all attended vectors and feed the decoder RNN with the result
            fusion = self.merge_op(txt_z_t)
            h = self.dec(fusion, h)
            z_states_list.append(h.unsqueeze(0))

        # store the states into a tensor so that the decoders can use it seemlessly
        z_states = torch.cat(z_states_list, 0)
        return z_states

    def _merge_sum(self, att_ctx_dict):
        summ = None
        for e in att_ctx_dict.keys():
            if summ is None:
                summ = torch.autograd.Variable(torch.zeros(att_ctx_dict[e].shape)).cuda()
            summ += att_ctx_dict[e]
        return summ


