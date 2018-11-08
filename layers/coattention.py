# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoAttention(nn.Module):
    """Co-attention between two sequence.

    Use one hidden layer to compute an affinity matrix between two sequences.
    This can be then normalized in two direction which gives us 1->2 and 2->1
    attentions.

    The co-attention is computed using a single feed-forward layer as in
    Bahdanau's attention.
    """
    def __init__(self, ctx_1_dim, ctx_2_dim, bottleneck,
                 att_activ='tanh', mlp_bias=False):
        super().__init__()

        self.mlp_hid = nn.Conv2d(ctx_1_dim + ctx_2_dim, bottleneck, 1)
        self.mlp_out = nn.Conv2d(bottleneck, 1, 1, bias=mlp_bias)
        self.activ = getattr(F, att_activ)

        self.project_1_to_2 = nn.Linear(ctx_1_dim + ctx_2_dim, bottleneck)
        self.project_2_to_1 = nn.Linear(ctx_1_dim + ctx_2_dim, bottleneck)

    def forward(self, ctx_1, ctx_2, ctx_1_mask=None, ctx_2_mask=None):
        if ctx_2_mask is not None:
            ctx_2_neg_mask = (1. - ctx_2_mask.transpose(0, 1).unsqueeze(1)) * -1e12


        ctx_1_len = ctx_1.size(0)
        ctx_2_len = ctx_2.size(0)
        
        b_ctx_1 = ctx_1.permute(1, 2, 0).unsqueeze(3).repeat(1, 1, 1, ctx_2_len)
        b_ctx_2 = ctx_2.permute(1, 2, 0).unsqueeze(2).repeat(1, 1, ctx_1_len, 1)

        catted = torch.cat([b_ctx_1, b_ctx_2], dim=1)
        #concatenate the two context vectors
        hidden = self.activ(self.mlp_hid(catted))
        affinity_matrix = self.mlp_out(hidden).squeeze(1)
        if ctx_1_mask is not None:
            ctx_1_neg_mask = (1. - ctx_1_mask.transpose(0, 1).unsqueeze(2)) * -1e12
            affinity_matrix += ctx_1_neg_mask

        if ctx_2_mask is not None:
            ctx_2_neg_mask = (1. - ctx_2_mask.transpose(0, 1).unsqueeze(1)) * -1e12
            affinity_matrix += ctx_2_neg_mask

        dist_1_to_2 = F.softmax(affinity_matrix, dim=2)
        context_1_to_2 = ctx_1.permute(1, 2, 0).matmul(dist_1_to_2).permute(2, 0, 1)
        seq_1_to_2 = self.activ(
            self.project_1_to_2(torch.cat([ctx_2, context_1_to_2], dim=-1)))

        dist_2_to_1 = F.softmax(affinity_matrix, dim=1).transpose(1, 2)
        context_2_to_1 = ctx_2.permute(1, 2, 0).matmul(dist_2_to_1).permute(2, 0, 1)
        seq_2_to_1 = self.activ(
            self.project_2_to_1(torch.cat([ctx_1, context_2_to_1], dim=-1)))

        return seq_2_to_1, seq_1_to_2


class MultiHeadCoAttention(nn.Module):
    """Generalization of multi-head attention for co-attention."""

    def __init__(self, ctx_1_dim, ctx_2_dim, bottleneck, head_count, dropout=0.1):
        assert bottleneck % head_count == 0
        self.dim_per_head = bottleneck // head_count
        self.model_dim = bottleneck

        super().__init__()
        self.head_count = head_count

        self.linear_keys_1 = nn.Linear(ctx_1_dim,
                                       head_count * self.dim_per_head)
        self.linear_values_1 = nn.Linear(ctx_1_dim,
                                         head_count * self.dim_per_head)
        self.linear_keys_2 = nn.Linear(ctx_2_dim,
                                       head_count * self.dim_per_head)
        self.linear_values_2 = nn.Linear(ctx_2_dim,
                                         head_count * self.dim_per_head)

        self.final_1_to_2_linear = nn.Linear(bottleneck, bottleneck)
        self.final_2_to_1_linear = nn.Linear(bottleneck, bottleneck)
        self.project_1_to_2 = nn.Linear(ctx_1_dim + ctx_2_dim, bottleneck)
        self.project_2_to_1 = nn.Linear(ctx_1_dim + ctx_2_dim, bottleneck)


    def forward(self, ctx_1, ctx_2, ctx_1_mask=None, ctx_2_mask=None):
        """
        Compute the context vector and the attention vectors.
        """

        batch_size = ctx_1.size(1)
        assert batch_size == ctx_2.size(1)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        ctx_1_len = ctx_1.size(0)
        ctx_2_len = ctx_2.size(0)

        def shape(x, length):
            """  projection """
            return x.view(
                length, batch_size, head_count, dim_per_head).permute(1, 2, 0, 3)

        def unshape(x, length):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(
                batch_size, length, head_count * dim_per_head).transpose(0, 1)

        # 1) Project key, value, and key_2.
        key_1_up = shape(self.linear_keys_1(ctx_1), ctx_1_len)
        value_1_up = shape(self.linear_values_1(ctx_1), ctx_1_len)
        key_2_up = shape(self.linear_keys_2(ctx_2), ctx_2_len)
        value_2_up = shape(self.linear_values_2(ctx_2), ctx_2_len)

        scores = torch.matmul(key_2_up, key_1_up.transpose(2, 3))
        
        #Padding the "scores" if required
        if ctx_1_mask is not None:
            mask = ctx_1_mask.t().unsqueeze(2).unsqueeze(3).expand_as(scores)
            scores = scores.masked_fill(mask.byte(), -1e18)
        if ctx_2_mask is not None:
            mask = ctx_2_mask.t().unsqueeze(1).unsqueeze(3).expand_as(scores)
            scores = scores.masked_fill(mask.byte(), -1e18)

        # 3) Apply attention dropout and compute context vectors.
        dist_1_to_2 = F.softmax(scores, dim=2) #Get attention weights
        context_1_to_2 = unshape(torch.matmul(dist_1_to_2, value_1_up), ctx_2_len)
        context_1_to_2 = self.final_1_to_2_linear(context_1_to_2)
        seq_1_to_2 = self.activ(
            self.project_1_to_2(torch.cat([ctx_2, context_1_to_2], dim=-1)))
            #Assuming seq_1_to_2 is attention vector since tanh is applied to it

        cov_loss=0.0
        cov_loss+= torch.sum (torch.min (seq_1_to_2,coverage) )

        # 3.3. Update coverage
        coverage+= seq_1_to_2


        dist_2_to_1 = F.softmax(scores, dim=1)
        context_2_to_1 = unshape(
            torch.matmul(dist_2_to_1.transpose(2, 3), value_2_up), ctx_1_len)
        context_2_to_1 = self.final_2_to_1_linear(context_2_to_1)
        seq_2_to_1 = self.activ(
            self.project_2_to_1(torch.cat([ctx_1, context_2_to_1], dim=-1)))

        return context_2_to_1, context_1_to_2
