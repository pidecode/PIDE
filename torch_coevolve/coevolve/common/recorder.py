from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from collections import defaultdict
from coevolve.common.cmd_args import cmd_args
from copy import deepcopy

class DurDist(object):
    def __init__(self, buf_size):
        self.buf_size = buf_size
        self.pos = 0
        self.records = []
    
    def add_time(self, dur):
        if len(self.records) < self.buf_size:
            self.records.append(dur)
        else:
            self.records[self.pos] = dur
            self.pos = (self.pos + 1) % self.buf_size
    
    def print_dist(self):
        m = np.mean(self.records)
        max = np.max(self.records)
        min = np.min(self.records)
        print('dur dist: [%.2f, %.2f] with mean %.2f' % (min, max, m))

dur_dist = DurDist(10000)


class Recorder(object):
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
    
    def update_event(self, user, item, t):
        if user in self.cur_user_time:
            assert t >= self.cur_user_time[user]
        if item in self.cur_item_time:
            assert t >= self.cur_item_time[item]
        
        self.cur_user_time[user] = t
        self.cur_item_time[item] = t
        self.user_item_time[user][item] = t
    
    def get_cur_time(self, user, item):
        t = self.t_begin
        if user in self.cur_user_time:
            t = max(t, self.cur_user_time[user])
        if item in self.cur_item_time:
            t = max(t, self.cur_item_time[item])
        return t
    
    def get_last_interact_time(self, user, item):
        if item in self.user_item_time[user]:
            return self.user_item_time[user][item]
        return -1000000

    def dump(self):
        return self.t_begin, deepcopy(self.cur_user_time), deepcopy(self.cur_item_time), deepcopy(self.user_item_time)

    def load_dump(self, t_begin, cur_user_time, cur_item_time, user_item_time):
        self.t_begin = t_begin
        self.cur_user_time = deepcopy(cur_user_time)
        self.cur_item_time = deepcopy(cur_item_time)
        self.user_item_time = deepcopy(user_item_time)

    def reset(self, t_begin):
        self.t_begin = t_begin

        self.cur_user_time = {}
        self.cur_item_time = {}
        self.user_item_time = defaultdict(dict)

cur_time = Recorder(cmd_args.num_users, cmd_args.num_items)