from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import random
from collections import defaultdict
from coevolve.common.cmd_args import cmd_args


class Edge(object):
    def __init__(self, edge_idx, dst):
        self.edge_idx = edge_idx
        self.dst = dst

class BipartiteGraph(object):
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.reset()

    def reset(self):
        self.user_list = []
        self.item_list = []
        self.user_idx = {}
        self.item_idx = {}
        self.edge_list = []

        self.user_edge_list = defaultdict(list)
        self.user_neighbors = defaultdict(set)

        self.item_edge_list = defaultdict(list)
        self.item_neighbors = defaultdict(set)

    def add_event(self, user, item):
        if not user in self.user_idx:
            idx = len(self.user_idx)
            self.user_idx[user] = idx
            self.user_list.append(user)

        if not item in self.item_idx:
            idx = len(self.item_idx)
            self.item_idx[item] = idx
            self.item_list.append(item)

        if item in self.user_neighbors[user]:
            return
        
        edge_idx = len(self.edge_list)
        self.edge_list.append((user, item))
        
        self.user_edge_list[user].append(Edge(edge_idx, item))
        self.user_neighbors[user].add(item)
        self.item_edge_list[item].append(Edge(edge_idx, user))
        self.item_neighbors[item].add(user)

bg = BipartiteGraph(cmd_args.num_users, cmd_args.num_items)