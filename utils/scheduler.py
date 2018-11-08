# -*- coding: utf-8 -*-
import logging
import pdb

import torch
import random
from itertools import chain, combinations, cycle

from .topology import Topology

logger = logging.getLogger('nmtpytorch')

class Scheduler(object):

    def __init__(self, topology, schedule_type_enc=None, schedule_type_dec=None,
                droptask_prob=1, droptask_e_delay=None, manual_schedule=None):

        self.topology = topology
        self.schedule_type_enc = schedule_type_enc
        self.schedule_type_dec = schedule_type_dec
        self.droptask_prob = droptask_prob
        self.droptask_e_delay = droptask_e_delay
        self.manual_schedule = manual_schedule

        if self.manual_schedule is not None:
            if not isinstance(self.manual_schedule, dict):
                raise RuntimeError("'{}' not recognized for manual_schedule. Use a dict with form {'id': 'direction'@'num_batches'}".format(self.manual_schedule))
            for k,v in self.manual_schedule.items():
                try:
                    self.manual_schedule[k] = [half.strip() for half in v.split('@')]
                    self.manual_schedule[k][0] = Topology(self.manual_schedule[k][0])
                    self.manual_schedule[k][1] = int(self.manual_schedule[k][1])
                except:
                    logger.info("Incorrect format for manual_schedule. Use a dict with form {0 : En1 -> En2, Pt2 @ 50, 1 : Pt1 -> Pt2, En2 @ 50, ... }, for example.")
            # Manual scheduling initializations:
            # keep track internally of how many batches the scheduler has seen
            self.batch_ctr = 0                                  # how many batches have been seen since last reset
            self.key_cyc = cycle(self.manual_schedule.keys())   # cycle of possible encoder/decoder setup options
            self.curr_key = next(self.key_cyc)                       # key for which encoder/decoder setup is currently being used
            self.enc_ids = self.manual_schedule[self.curr_key][0].srcs
            self.dec_ids = self.manual_schedule[self.curr_key][0].trgs
        else:
            self.enc_ids = self.topology.srcs
            self.dec_ids = self.topology.trgs

    # Utility function for generating the modified powerset of randomization options:
    # e.g. powerset([1,2,3]) -> () (1,) (2,) (3,) (1,2) (1,3) (2,3)  (don't alllow all to be dropped!)
    def powerset(opts):
        chain.from_iterable(combinations(opts, r) for r in range(len(opts)-1))

    def _inc_counter(self):
        self.batch_ctr += 1
        if self.batch_ctr > self.manual_schedule[self.curr_key][1]:
            self.curr_key = next(self.key_cyc)
            self.batch_ctr = 0
            self.enc_ids = self.manual_schedule[self.curr_key][0].srcs
            self.dec_ids = self.manual_schedule[self.curr_key][0].trgs

    def get_encs_and_decs(self):
        return self._get_encoders(), self._get_decoders()


    def _get_encoders(self):
        """Performs droptask for encoders.
        Arguments:
        Returns:
            list: A list of keys for which encoders to apply.
        """
        # If a manual schedule for a batch-level regime is given, use it
        # Manual schedules account for both encoder and decoders (all other params can be ignored)
        if self.manual_schedule is not None:

            # increment batch counter and select the current scheduled task
            self._inc_counter()

            these_encs = self.manual_schedule[self.curr_key][0].srcs
            #logger.info('Scheduler: batch_ctr is {}, curr_key is {} \n these_encs: {}\n these_decs: {}'.format(self.batch_ctr, self.curr_key, these_encs, these_decs))
            # return appropriate set of encoders
            return these_encs

        # Otherwise, do some kind of random droptask
        these_encoders = self.enc_ids.copy()
        droptask = self.schedule_type_enc

        if droptask is not None:
            # Do random droptask only with specified probability
            if random.uniform(0, 1) < self.droptask_prob:
                # Sample a random subset of encoder(s) to drop from contributing to z
                if droptask == 'random':
                    drop_choices = list(self.powerset(self.enc_ids.keys()))
                elif droptask == 'random_1':
                    drop_choices = list(self.enc_ids.keys())
                else:
                    raise Exception("Scheduler: Encoder droptask scheduler option '{}' is unknown. Use (None|random|random_1)".format(droptask))
                for c in drop_choices[random.randint(0, len(drop_choices)-1)]:
                    #logger.info("Scheduler: dropping {} '{}'".format(which_droptask, c))
                    del these_encoders[c]
        return list(these_encoders.keys())

    def _get_decoders(self):
        """Performs droptask for decoders.
        Arguments:
        Returns:
            list: A list of keys for which decoders to apply.
        """
        # If a manual schedule for a batch-level regime is given, use it
        # Manual schedules account for both encoder and decoders (all other params can be ignored)
        if self.manual_schedule is not None:
            these_decs = self.manual_schedule[self.curr_key][0].trgs
            #logger.info('Scheduler: batch_ctr is {}, curr_key is {} \n these_encs: {}\n these_decs: {}'.format(self.batch_ctr, self.curr_key, these_encs, these_decs))
            # return appropriate set of decoders
            return these_decs

        # Otherwise, do some kind of random droptask
        these_decoders = self.dec_ids.copy()
        droptask = self.schedule_type_dec

        if droptask is not None:
            # Do random droptask only with specified probability
            if random.uniform(0, 1) < self.droptask_prob:
                # Sample a random subset of encoder(s) to drop from contributing to z
                if droptask == 'random':
                    drop_choices = list(self.powerset(self.decs.keys()))
                elif droptask == 'random_1':
                    drop_choices = list(self.decs.keys())
                else:
                    raise Exception("Scheduler: Decoders droptask scheduler option '{}' is unknown. Use (None|random|random_1)".format(droptask))
                for c in drop_choices[random.randint(0, len(drop_choices)-1)]:
                    #logger.info("Scheduler: dropping {} '{}'".format(which_droptask, c))
                    del these_decoders[c]
        return list(these_decoders.keys())




