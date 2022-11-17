from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
from scipy.stats import rankdata
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from coevolve.common.recorder import cur_time, dur_dist
from coevolve.common.consts import DEVICE
from coevolve.common.cmd_args import cmd_args
from coevolve.model.rayleigh_proc import ReyleighProc
from coevolve.common.neg_sampler import rand_sampler

from coevolve.common.pytorch_utils import SpEmbedding, weights_init
import networkx as nx

from coevolve.common.dataset import merge_list, create_kg, kg_add, get_neigbors


class DeepCoevolve(nn.Module):
    def __init__(self, train_data, test_data, num_users, num_items, embed_size, score_func, dt_type, max_norm):
        super(DeepCoevolve, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.max_norm = max_norm
        self.embed_size = embed_size
        self.score_func = score_func
        self.dt_type = dt_type
        self.user_embedding = SpEmbedding(num_users, embed_size)
        self.item_embedding = SpEmbedding(num_items, embed_size)

        self.user_lookup_embed = {}
        self.item_lookup_embed = {}
        self.delta_t = np.zeros((self.num_items,), dtype=np.float32)
        self.user_cell = nn.GRUCell(embed_size, embed_size)
        self.item_cell = nn.GRUCell(embed_size, embed_size)

        self.train_data = train_data
        self.test_data = test_data
        self.inter_linear = nn.Linear(embed_size, embed_size)
        self.neibor_linear = nn.Linear(embed_size, embed_size)
        self.train_user_list = train_data.user_list
        self.train_item_list = train_data.item_list
        self.test_user_list = test_data.user_list
        self.test_item_list = test_data.item_list
        self.user_list = merge_list(self.train_user_list, self.test_user_list)
        self.item_list = merge_list(self.train_item_list, self.test_item_list)
        self.kg = nx.Graph()

    def normalize(self):
        if self.max_norm is None:
            return
        self.user_embedding.normalize(self.max_norm)
        self.item_embedding.normalize(self.max_norm)

    def _get_embedding(self, side, idx, lookup):
        if not idx in lookup:
            if side == 'user':
                lookup[idx] = self.user_embedding([idx])
            else:
                lookup[idx] = self.item_embedding([idx])
        return lookup[idx]

    def get_cur_user_embed(self, user):
        return self._get_embedding('user', user, self.user_lookup_embed)

    def get_cur_item_embed(self, item):
        return self._get_embedding('item', item, self.item_lookup_embed)

    def get_pred_score(self, comp, delta_t):
        if self.score_func == 'log_ll':
            d_t = np.clip(delta_t, a_min=1e-10, a_max=None)
            return np.log(d_t) + np.log(comp) - 0.5 * comp * (d_t ** 2)
        elif self.score_func == 'comp':
            return comp
        elif self.score_func == 'intensity':
            return comp * delta_t
        else:
            raise NotImplementedError

    def get_output(self, cur_event, phase, kg, depth):
        cur_user_embed = self.get_cur_user_embed(cur_event.user)
        cur_item_embed = self.get_cur_item_embed(cur_event.item)

        neibor = get_neigbors(kg, cur_event.user, depth)
        for i in neibor[depth]:
            if i != cur_event.user:
                if i in self.user_list:
                    self._get_embedding('user', i, self.user_lookup_embed)
                if i in self.item_list:
                    self._get_embedding('item', i, self.item_lookup_embed)

        neibor = get_neigbors(kg, cur_event.item, depth)
        for j in neibor[depth]:
            if j != cur_event.item:
                if j in self.user_list:
                    self._get_embedding('user', j, self.user_lookup_embed)
                if j in self.item_list:
                    self._get_embedding('item', j, self.item_lookup_embed)

        t_end = cur_event.t
        base_comp = ReyleighProc.base_compatibility(cur_user_embed, cur_item_embed)#计算intensity function的过程

        dur = cur_event.t - cur_time.get_cur_time(cur_event.user, cur_event.item)
        dur_dist.add_time(dur)
        time_pred = ReyleighProc.time_mean(base_comp)

        mae = torch.abs(time_pred - dur)
        mse = (time_pred - dur) ** 2

        if phase == 'test':
            comp = ReyleighProc.base_compatibility(cur_user_embed, self.updated_item_embed).view(-1).cpu().data.numpy()
            for i in range(self.num_items):
                prev = cur_time.get_last_interact_time(cur_event.user,
                                                       i) if self.dt_type == 'last' else cur_time.get_cur_time(
                    cur_event.user, i)
                self.delta_t[i] = cur_event.t - prev
            scores = self.get_pred_score(comp, self.delta_t)
            ranks = rankdata(-scores)
            mar = ranks[cur_event.item]
            return mar, mae, mse
        neg_users = rand_sampler.sample_neg_users(cur_event.user, cur_event.item)
        neg_items = rand_sampler.sample_neg_items(cur_event.user, cur_event.item)

        neg_users_embeddings = self.user_embedding(neg_users)
        neg_items_embeddings = self.item_embedding(neg_items)
        for i, u in enumerate(neg_users):
            if u in self.user_lookup_embed:
                neg_users_embeddings[i] = self.user_lookup_embed[u]
        for j, i in enumerate(neg_items):
            if i in self.item_lookup_embed:
                neg_items_embeddings[j] = self.item_lookup_embed[i]
        #计算survive function值
        survival = ReyleighProc.survival(cur_event.user, cur_user_embed,
                                         cur_event.item, cur_item_embed,
                                         neg_users_embeddings, neg_users,
                                         neg_items_embeddings, neg_items,
                                         cur_event.t)
        loss = -torch.log(base_comp) + survival
        return loss, mae, mse

    def forward(self, T_begin, events, phase):

        self.kg.clear()
        kg = create_kg(self.kg, T_begin, self.train_data.user_event_lists, self.train_data.item_event_lists)

        if phase == 'train':
            cur_time.reset(T_begin)
        self.user_lookup_embed = {}
        self.item_lookup_embed = {}

        with torch.set_grad_enabled(phase == 'train'):
            if phase == 'test':
                self.updated_user_embed = self.user_embedding.weight.clone()
                self.updated_item_embed = self.item_embedding.weight.clone()

            loss = 0.0
            mae = 0.0
            mse = 0.0
            pbar = enumerate(events)

            for e_idx, cur_event in pbar:
                depth = 0
                user_user_neibor = []
                user_item_neibor = []
                item_user_neibor = []
                item_item_neibor = []
                assert cur_event.t >= T_begin

                kg_add(kg, cur_event.user, cur_event.item, cur_event.t)
                if e_idx==95:
                    flagto=0
                cur_loss, cur_mae, cur_mse = self.get_output(cur_event, phase, kg, 1)
                loss += cur_loss
                mae += cur_mae
                mse += cur_mse
                if e_idx + 1 == len(events):
                    break
                # coevolve的部分，更新交互节点
                cur_user_embed = self.user_lookup_embed[cur_event.user]
                cur_item_embed = self.item_lookup_embed[cur_event.item]
                self.user_lookup_embed[cur_event.user] = self.user_cell(cur_item_embed, cur_user_embed)
                self.item_lookup_embed[cur_event.item] = self.item_cell(cur_user_embed, cur_item_embed)

                #影响扩散的部分
                depth = 1
                neibor = get_neigbors(kg, cur_event.user, depth)#获取user节点的邻居
                if depth in neibor:
                    for n in neibor[depth]:
                        if n in self.user_list:
                            user_user_neibor.append(n)
                        if n in self.item_list:
                            user_item_neibor.append(n)
                for i in user_user_neibor:
                    if i != cur_event.item:
                        neibor_embedding = self.user_embedding([i])#获取邻居节点嵌入
                        inter_sim = neibor_embedding * cur_user_embed#点乘
                        inter_sim = self.inter_linear(inter_sim)
                        neibor_embedding = self.neibor_linear(neibor_embedding)
                        neiborghhood_embedding = neibor_embedding + inter_sim
                        self.user_lookup_embed[i] = nn.Sigmoid()(neiborghhood_embedding)#激活函数
                for i in user_item_neibor:
                    if i != cur_event.user:
                        neibor_embedding = self.item_embedding([i])
                        inter_sim = neibor_embedding * cur_user_embed
                        inter_sim = self.inter_linear(inter_sim)
                        neibor_embedding = self.neibor_linear(neibor_embedding)
                        neiborghhood_embedding = neibor_embedding + inter_sim
                        self.item_lookup_embed[i] = nn.Sigmoid()(neiborghhood_embedding)
                neibor = get_neigbors(kg, cur_event.item, depth)
                if depth in neibor:
                    for n in neibor[depth]:
                        if n in self.user_list:
                            item_user_neibor.append(n)
                        if n in self.item_list:
                            item_item_neibor.append(n)
                for i in item_user_neibor:
                    if i != cur_event.item:
                        neibor_embedding = self.user_embedding([i])
                        inter_sim = neibor_embedding * cur_user_embed
                        inter_sim = self.inter_linear(inter_sim)
                        neibor_embedding = self.neibor_linear(neibor_embedding)
                        neiborghhood_embedding = neibor_embedding + inter_sim
                        self.user_lookup_embed[i] = nn.Sigmoid()(neiborghhood_embedding)
                for i in item_item_neibor:
                    if i != cur_event.user:
                        neibor_embedding = self.item_embedding([i])
                        inter_sim = neibor_embedding * cur_user_embed
                        inter_sim = self.inter_linear(inter_sim)
                        neibor_embedding = self.neibor_linear(neibor_embedding)
                        neiborghhood_embedding = neibor_embedding + inter_sim
                        self.item_lookup_embed[i] = nn.Sigmoid()(neiborghhood_embedding)
                #



                if phase == 'test':  # update embeddings into the embed mat
                    self.updated_user_embed[cur_event.user] = self.user_lookup_embed[cur_event.user]
                    self.updated_item_embed[cur_event.item] = self.item_lookup_embed[cur_event.item]
                cur_time.update_event(cur_event.user, cur_event.item, cur_event.t)

            rmse = torch.sqrt(mse / len(events)).item()
            mae = mae.item() / len(events)
            torch.set_grad_enabled(True)
            if phase == 'train':
                return loss, mae, rmse
            else:
                return loss / len(events), mae, rmse


def del_list_element(list,element):
    for i in list:
        if list[i]==element:
            del list[i]
    return list

