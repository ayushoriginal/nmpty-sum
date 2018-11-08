# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import torch.nn as nn

from ..layers.speech import BiRNNPv1
from ..layers import SwitchingGRUDecoder, TextEncoder
from ..utils.misc import get_n_params
from ..utils.nn import ModuleDict
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')


class ASRNMT(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'te_enc_dim': 256,          # Text encoder dim
            'te_emb_dim': 128,          # Text embedding dim
            'te_enc_type': 'gru',       # Text encoder type
            'te_n_encoders': 2,         # Text number of layer
            'sp_feat_dim': 43,          # Speech features dimensionality
            'sp_enc_dim': 256,          # Speech encoder dim
            'sp_enc_type': 'gru',       # Speech encoder type
            'sp_enc_subsample': (),     # Tuple of subsampling factors
                                        # Also defines # of subsampling layers
            'sp_n_sub_layers': 1,       # Number of stacked RNNs in each subsampling block
            'sp_n_base_encoders': 1,    # Number of stacked encoders
            'trg_emb_dim': 128,         # Decoder embedding dim
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            'dec_init_size': None,      # feature vector dimensionality for
                                        # dec_init == 'feats'
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'scheduling': 'random',     # Task scheduling type (random|alternating)
            'scheduling_p': (0.5, 0.5), # Scheduling probabilities for encoders for 'random'
            'dropout': 0,               # Generic dropout overall the architecture
            'tied_dec_embs': False,     # Share decoder embeddings
            'max_len': None,            # Reject samples if len('bucket_by') > max_len
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Can be 'ascending' or 'descending' to train
                                        # with increasing/decreasing sizes of sequences
            'direction': None,          # Network directionality, i.e. en->de
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Vocabulary objects
        self.vocabs = {}

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Setup options
        self.opts.model = self.set_model_options(opts.model)

        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, name=name)

        self.speech = str(self.topology.get_srcs('Kaldi')[0])
        self.sl = str(self.topology.get_srcs('Text')[0])
        self.src_vocab = self.vocabs[self.sl]
        self.n_src_vocab = len(self.src_vocab)

        # Many to one architecture: 1 decoder
        self.tl = self.topology.get_trgs('Text')[0]
        self.trg_vocab = self.vocabs[self.tl]
        self.n_trg_vocab = len(self.trg_vocab)

        # Need to be set for early-stop evaluation
        # NOTE: This should come from config or elsewhere
        self.val_refs = self.opts.data['val_set'][self.tl]

        # Pick the correct encoder sampler method
        self.get_training_encoder = getattr(
            self, '_get_{}_encoder'.format(self.opts.model['scheduling']))
        self.scheduling_p = self.opts.model['scheduling_p']
        self.task_order = []

    def __repr__(self):
        s = super().__repr__() + '\n'
        for vocab in self.vocabs.values():
            s += "{}\n".format(vocab)
        s += "{}\n".format(get_n_params(self))
        return s

    def set_model_options(self, model_opts):
        self.set_defaults()
        for opt, value in model_opts.items():
            if opt in self.defaults:
                # Override defaults from config
                self.defaults[opt] = value
            else:
                logger.info('Warning: unused model option: {}'.format(opt))
        return self.defaults

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:
                nn.init.kaiming_normal(param.data)

    def setup(self, is_train=True):
        speech_enc = BiRNNPv1(
            input_size=self.opts.model['sp_feat_dim'],
            hidden_size=self.opts.model['sp_enc_dim'],
            rnn_type=self.opts.model['sp_enc_type'],
            dropout=self.opts.model['dropout'],
            subsample=self.opts.model['sp_enc_subsample'],
            num_sub_layers=self.opts.model['sp_n_sub_layers'],
            num_base_layers=self.opts.model['sp_n_base_encoders'])

        text_enc = TextEncoder(
            input_size=self.opts.model['te_emb_dim'],
            hidden_size=self.opts.model['te_enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['te_enc_type'],
            dropout_emb=self.opts.model['dropout'],
            dropout_ctx=self.opts.model['dropout'],
            dropout_rnn=self.opts.model['dropout'],
            num_layers=self.opts.model['te_n_encoders'])

        self.encoders = ModuleDict({
            self.speech: speech_enc,
            self.sl: text_enc,
        })

        # For attention
        self.modality_dict = {
            self.speech: (self.opts.model['sp_enc_dim'] * 2, 'mlp'),
            self.sl: (self.opts.model['te_enc_dim'] * 2, 'mlp'),
        }

        self.encoder_names = list(self.encoders.keys())
        self.n_encoders = len(self.encoder_names)

        ################
        # Create Decoder
        ################
        self.dec = SwitchingGRUDecoder(
            input_size=self.opts.model['trg_emb_dim'],
            modality_dict=self.modality_dict,
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            tied_emb=self.opts.model['tied_dec_embs'],
            dropout_out=self.opts.model['dropout'])

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'])
        logger.info(dataset)
        return dataset

    def get_bos(self, batch_size):
        """Returns a representation for <bos> embeddings for decoding."""
        return torch.LongTensor(batch_size).fill_(self.trg_vocab['<bos>'])

    def encode(self, batch, **kwargs):
        # Get the requested encoder
        enc_name = next(iter(kwargs['enc_ids']))
        enc = self.encoders[enc_name]
        return {str(enc_name): enc(batch[enc_name])}

    def _get_random_encoder(self, **kwargs):
        """Returns a random encoder with uniform probability by default."""
        if not any(self.task_order):
            # Buffer 10K samples for random scheduling
            order = np.random.choice(
                len(self.encoders), 10000, True, p=self.scheduling_p)
            self.task_order = [self.encoder_names[i] for i in order]

        # Return an encoder name
        return self.task_order.pop()

    def _get_alternating_encoder(self, **kwargs):
        """Returns the next encoder candidate based on update count."""
        return self.encoder_names[kwargs['uctr'] % self.n_encoders]

    def forward(self, batch, **kwargs):
        enc_ids = kwargs.get('enc_ids', None)
        if enc_ids is None and self.training:
            enc_ids = self.get_training_encoder(**kwargs)

        # Encode, decode, get loss and normalization factor
        result = self.dec(self.encode(batch, enc_ids=[enc_ids]), batch[self.tl])
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]
        return result

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss_dict = {k: Loss() for k in self.encoder_names}
        metrics = []

        for batch in data_loader:
            batch.to_gpu(volatile=True)
            for enc_id, loss in loss_dict.items():
                out = self.forward(batch, enc_ids=enc_id)
                loss.update(out['loss'], out['n_items'])

        for enc_id, loss in loss_dict.items():
            loss = loss.get()
            logger.info('Task {} loss: {:.3f}'.format(enc_id, loss))
            metrics.append(loss)

        return [Metric('LOSS', np.mean(metrics), higher_better=False)]

    def get_decoder(self, task_id=None):
        return self.dec
