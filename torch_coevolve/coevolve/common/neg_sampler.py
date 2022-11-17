from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import random
from coevolve.common.cmd_args import cmd_args


class RandSampler(object):
    def __init__(self, num_users, num_items, neg_users, neg_items):
        self.num_users = num_users
        self.num_items = num_items

        self.neg_user_gen = self.sampling_gen(num_users, neg_users)
        self.neg_item_gen = self.sampling_gen(num_items, neg_items)

    def sample_neg_items(self, user, item):
        ids = next(self.neg_item_gen)
        if not item in ids:
            ids[0] = item
        return ids
    
    def sample_neg_users(self, user, item):
        ids = next(self.neg_user_gen)
        if not user in ids:
            ids[0] = user
        return ids

    def sampling_gen(self, n_samples, num):
        indices = list(range(num))

        while True:
            random.shuffle(indices)

            for i in range(0, len(indices), num):
                if i + num > len(indices):
                    break                
                yield indices[i:i+num]


rand_sampler = RandSampler(cmd_args.num_users, cmd_args.num_items,
                           cmd_args.neg_users, cmd_args.neg_items)