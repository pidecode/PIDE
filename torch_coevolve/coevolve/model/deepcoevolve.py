from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from math import ceil
import math
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
import random
from coevolve.common.dataset import merge_list, create_kg, kg_add, get_neigbors, threshold, softmax, removeEmptyDict


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
        self.W_1_1 = nn.Linear(embed_size, embed_size)
        self.W_1_2 = nn.Linear(embed_size, embed_size)
        self.V_1_1 = nn.Linear(embed_size, embed_size)
        self.V_1_2 = nn.Linear(embed_size, embed_size)
        self.W_2_1 = nn.Linear(embed_size, embed_size)
        self.W_2_2 = nn.Linear(embed_size, embed_size)
        self.V_2_1 = nn.Linear(embed_size, embed_size)
        self.V_2_2 = nn.Linear(embed_size, embed_size)
        self.W_3_1 = nn.Linear(embed_size, embed_size)
        self.W_3_2 = nn.Linear(embed_size, embed_size)
        self.V_3_1 = nn.Linear(embed_size, embed_size)
        self.V_3_2 = nn.Linear(embed_size, embed_size)
        self.W_4_1 = nn.Linear(embed_size, embed_size)
        self.W_4_2 = nn.Linear(embed_size, embed_size)
        self.V_4_1 = nn.Linear(embed_size, embed_size)
        self.V_4_2 = nn.Linear(embed_size, embed_size)
        self.train_user_list = train_data.user_list
        self.train_item_list = train_data.item_list
        self.test_user_list = test_data.user_list
        self.test_item_list = test_data.item_list
        self.user_list = merge_list(self.train_user_list, self.test_user_list)
        self.item_list = merge_list(self.train_item_list, self.test_item_list)
        self.time_matrix=[[-1 for i in range(self.num_items)] for i in range(self.num_users)]
        self.relation_matrix=[[-1 for i in range(self.num_items)] for i in range(self.num_users)]
        self.kg = nx.Graph()
        # self.drop_ration=0.2
        self.user_ground={}
        self.neighbor_ration=0
        self.time_threshold=0
        self.transForm1 = nn.Linear(in_features=2*embed_size,out_features=embed_size)
        self.transForm2 = nn.Linear(in_features=embed_size,out_features=32)
        self.transForm3 = nn.Linear(in_features=32,out_features=1)
        self.transForm1_2 = nn.Linear(in_features=2*embed_size,out_features=embed_size)
        self.transForm2_2 = nn.Linear(in_features=embed_size,out_features=32)
        self.transForm1_3 = nn.Linear(in_features=2 * embed_size, out_features=embed_size)
        self.transForm2_3 = nn.Linear(in_features=embed_size, out_features=32)
        self.transForm1_4 = nn.Linear(in_features=2 * embed_size, out_features=embed_size)
        self.transForm2_4 = nn.Linear(in_features=embed_size, out_features=32)


        self.tras_reduce=nn.Sequential(
            nn.Linear(in_features=2*embed_size,out_features=embed_size),
            nn.Linear(in_features=embed_size,out_features=32),
        )

        self.trans = nn.Sequential(
            nn.Linear(in_features=2*embed_size,out_features=embed_size),
            nn.Linear(in_features=embed_size,out_features=32),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

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

    def get_output(self,t_interaction,cur_event, phase, kg, depth,user_rec_5,user_rec_10,user_rec_20):
        cur_user_embed = self.get_cur_user_embed(cur_event.user)
        cur_item_embed = self.get_cur_item_embed(cur_event.item)
        if cur_event.user not in self.user_ground:
            self.user_ground[cur_event.user]=[]
        if cur_event.item not in self.user_ground[cur_event.user]:
            self.user_ground[cur_event.user].append(cur_event.item)
        user_set=[]
        item_set=[]
        interaction_set=[]
        one_hop=[]
        two_hop = []
        three_hop=[]
        four_hop=[]

        for event in t_interaction:
            ur = 'u' + str(event.user)
            im = 'i' + str(event.item)
            user_set.append(ur)
            item_set.append(im)

            if cmd_args.hop >= 1:
                interaction_set.extend([ur, im])#1-hop
                for i in interaction_set:
                    one_hop.extend(get_neigbors(kg, i, 1)[1])
                one_hop = list(set(one_hop))  # 删除相同元素
                one_hop = del_list_element(one_hop, interaction_set)
                for i in one_hop:
                    if i[0]=='u':user_set.append(i)
                    if i[0]=='i':item_set.append(i)

            if cmd_args.hop >= 2:
                for i in one_hop:#2-hop'
                    two_hop.extend(get_neigbors(kg, i, 1)[1])
                two_hop = list(set(two_hop))  # 删除相同元素
                two_hop = del_list_element(two_hop, interaction_set)
                two_hop = del_list_element(two_hop, one_hop)
                for i in two_hop:
                    if i[0]=='u':user_set.append(i)
                    if i[0]=='i':item_set.append(i)
            if cmd_args.hop >= 3:
                for i in two_hop:#3-hop
                    three_hop.extend(get_neigbors(kg, i, 1)[1])
                three_hop = list(set(three_hop))  # 删除相同元素
                three_hop = del_list_element(three_hop, interaction_set)
                three_hop = del_list_element(three_hop, two_hop)
                for i in three_hop:
                    if i[0]=='u':user_set.append(i)
                    if i[0]=='i':item_set.append(i)
            #
            if cmd_args.hop == 4:
                for i in three_hop:#4-hop
                    four_hop.extend(get_neigbors(kg, i, 1)[1])
                four_hop = list(set(four_hop))  # 删除相同元素
                four_hop = del_list_element(four_hop, interaction_set)
                four_hop = del_list_element(four_hop, three_hop)
                for i in four_hop:
                    if i[0]=='u':user_set.append(i)
                    if i[0]=='i':item_set.append(i)

        user_set = list(set(user_set))
        item_set = list(set(item_set))
        for i in user_set:
            self._get_embedding('user', int(i[1:]), self.user_lookup_embed)
        for i in item_set:
            self._get_embedding('item', int(i[1:]), self.item_lookup_embed)

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
            user_rec_5 = user_topK(5, user_rec_5, cur_event, ranks)
            user_rec_10 = user_topK(10, user_rec_10, cur_event, ranks)
            user_rec_20 = user_topK(20, user_rec_20, cur_event, ranks)
            return mar, mae, mse, user_rec_5, user_rec_10, user_rec_20,user_set,item_set

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
        return loss, mae, mse,user_set,item_set


    def forward(self, T_begin, events, phase):
        user_rec_5 = {}
        user_rec_10 = {}
        user_rec_20 = {}
        self.kg.clear()
        self.time_matrix = [[-1 for i in range(self.num_items)] for i in range(self.num_users)]
        kg = create_kg(self.kg,self.time_matrix,T_begin, self.train_data.user_event_lists, self.train_data.item_event_lists)
        T=self.train_data.ordered_events[0].t

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
            rank=[]
            pbar = enumerate(events)
            events_len=len(events)
            e_idx=0
            # for e_idx, cur_event in pbar:
            while e_idx<events_len:
                num_interaction=1
                t_interaction=[]
                # t_interaction.append(events[e_idx])
                # while events[e_idx].t==events[e_idx+num_interaction].t:
                #     t_interaction.append(events[e_idx+num_interaction])
                #     num_interaction=num_interaction+1
                #     if e_idx+num_interaction>=events_len:
                #         break
                e_idx=e_idx+num_interaction
                # if e_idx >= len(events):
                #     continue
                t_interaction.append(events[e_idx])


                cur_event = t_interaction[0]
                kg_reduce = kg
                kg_reduce = threshold(kg, self.time_threshold * cur_event.t)
                kg = kg_reduce

                for i in t_interaction:
                    ur = 'u' + str(i.user)
                    im = 'i' + str(i.item)
                    kg_add(kg_reduce, self.time_matrix, ur, im, i.t)

                cur_event=t_interaction[0]
                ur = 'u' + str(cur_event.user)
                im = 'i' + str(cur_event.item)
                assert cur_event.t >= T_begin

                if cur_event.user not in user_rec_5:
                    user_rec_5[cur_event.user]=[]
                if cur_event.user not in user_rec_10:
                    user_rec_10[cur_event.user]=[]
                if cur_event.user not in user_rec_20:
                    user_rec_20[cur_event.user]=[]



                if phase == 'test':
                    cur_loss, cur_mae, cur_mse,user_rec_5,user_rec_10,user_rec_20,user_set,item_set = self.get_output(t_interaction,cur_event, phase,kg_reduce, 1,user_rec_5,user_rec_10,user_rec_20)
                if phase == 'train':
                    cur_loss, cur_mae, cur_mse,user_set,item_set = self.get_output(t_interaction,cur_event, phase,kg_reduce, 1, user_rec_5,user_rec_10,user_rec_20)
                # cur_loss, cur_mae, cur_mse = self.get_output(cur_event, phase, kg, 1)
                if not math.isnan(cur_loss.item()):
                    loss += cur_loss
                    mae += cur_mae
                    mse += cur_mse
                if e_idx + 1 == len(events):
                    break
                # coevolve的部分，更新交互节点
                for cur_event in t_interaction:
                    cur_user_embed = self.user_lookup_embed[cur_event.user]
                    cur_item_embed = self.item_lookup_embed[cur_event.item]
                    self.user_lookup_embed[cur_event.user] = self.user_cell(cur_item_embed, cur_user_embed)
                    self.item_lookup_embed[cur_event.item] = self.item_cell(cur_user_embed, cur_item_embed)


                user_1_hop = {}
                user_1_hop_num=0
                item_1_hop = {}
                item_1_hop_num = 0
                time_list={}
                interaction_set=[]
                one_hop=[]
                one_hop_reduce=[]
                two_hop = []
                two_hop_reduce=[]
                three_hop = []
                three_hop_reduce=[]
                four_hop = []
                four_hop_reduce=[]
                one_hop_weight={}
                two_hop_weight={}
                three_hop_weight={}
                four_hop_weight={}
                one_hop_agg={}
                two_hop_agg = {}
                three_hop_agg = {}
                four_hop_agg = {}

                depth = 1
                for event in t_interaction:
                    ur = 'u' + str(event.user)
                    im = 'i' + str(event.item)
                    interaction_set.extend([ur,im])
                for i in interaction_set:
                    one_hop.extend(get_neigbors(kg, i, depth)[depth])
                one_hop=list(set(one_hop)) #删除相同元素
                one_hop=del_list_element(one_hop,interaction_set)


                for i in interaction_set:
                    one_hop_weight[i] = []
                    for j in one_hop:
                        if i[0]=='u' and j[0]=='i':
                            if self.time_matrix[int(i[1:])][int(j[1:])]!=-1:
                                one_hop_weight[i].append([self.time_matrix[int(i[1:])][int(j[1:])], j])
                        if i[0]=='i' and j[0]=='u':
                            if self.time_matrix[int(j[1:])][int(i[1:])]!=-1:
                                one_hop_weight[i].append([self.time_matrix[int(j[1:])][int(i[1:])], j])

                    one_hop_weight[i].sort(reverse=True)
                    neighbor_num=math.floor(len(one_hop_weight[i])*(1-self.neighbor_ration))
                    one_hop_weight[i]=one_hop_weight[i][:neighbor_num]
                    for n in one_hop_weight[i]:
                        if n[1] not in one_hop_reduce:
                            one_hop_reduce.append(n[1])
                    if one_hop_weight[i]==[]:
                        one_hop_weight.pop(i)
                one_hop=one_hop_reduce

                for i in one_hop_weight:
                    for j in one_hop_weight[i]:
                        if j[1] not in one_hop_agg:
                            one_hop_agg[j[1]]=[]
                        if i[0] == 'u' and j[1][0] == 'i':
                            one_hop_agg[j[1]].append([self.time_matrix[int(i[1:])][int(j[1][1:])], i])
                        if i[0] == 'i' and j[1][0] == 'u':
                            one_hop_agg[j[1]].append([self.time_matrix[int(j[1][1:])][int(i[1:])], i])
                for neighbor in one_hop_agg:#neighbor node
                    neighbor_node=int(neighbor[1:])
                    p = softmax(cur_event.t, one_hop_weight)
                    influence=0
                    if neighbor[0]=='u':
                        neibor_embedding = self.user_lookup_embed[neighbor_node]
                        neibor_embedding = self.V_1_1(neibor_embedding)
                    if neighbor[0]=='i':
                        neibor_embedding = self.item_lookup_embed[neighbor_node]
                        neibor_embedding = self.W_1_1(neibor_embedding)
                    for sources in one_hop_agg[neighbor]:#multiple source nodes
                        source_node=int(sources[1][1:])
                        if sources[1][0]=='u':
                            inter_sim = neibor_embedding * self.user_lookup_embed[source_node]
                            inter_sim = self.W_1_2(inter_sim)
                        if sources[1][0] == 'i':
                            inter_sim = neibor_embedding * self.item_lookup_embed[source_node]
                            inter_sim = self.V_1_2(inter_sim)
                        influence =influence+ p[neighbor][sources[1]] * inter_sim

                    if neighbor[0] == 'u': self.user_lookup_embed[neighbor_node] = nn.Sigmoid()(neibor_embedding+influence) # 激活函数
                    if neighbor[0] == 'i': self.item_lookup_embed[neighbor_node] = nn.Sigmoid()(neibor_embedding+influence) # 激活函数


###############################################################################
                if cmd_args.hop>=2:
                    depth = 2
                    for i in one_hop:
                        two_hop.extend(get_neigbors(kg, i, 1)[1])
                    two_hop = list(set(two_hop))  # 删除相同元素
                    two_hop = del_list_element(two_hop, interaction_set)
                    for i in one_hop:
                        two_hop_weight[i] = []
                        for j in two_hop:
                            if i[0] == 'u' and j[0] == 'i':
                                if self.time_matrix[int(i[1:])][int(j[1:])] != -1:
                                    two_hop_weight[i].append([self.time_matrix[int(i[1:])][int(j[1:])], j])
                            if i[0] == 'i' and j[0] == 'u':
                                if self.time_matrix[int(j[1:])][int(i[1:])] != -1:
                                    two_hop_weight[i].append([self.time_matrix[int(j[1:])][int(i[1:])], j])
                        two_hop_weight[i].sort(reverse=True)
                        neighbor_num = math.floor(len(two_hop_weight[i]) * (1 - self.neighbor_ration))
                        two_hop_weight[i] = two_hop_weight[i][:neighbor_num]
                        for n in two_hop_weight[i]:
                            if n[1] not in two_hop_reduce:
                                two_hop_reduce.append(n[1])
                        if two_hop_weight[i]==[]:
                            two_hop_weight.pop(i)
                    two_hop=two_hop_reduce
                    for i in two_hop_weight:
                        for j in two_hop_weight[i]:
                            if j[1] not in two_hop_agg:
                                two_hop_agg[j[1]] = []
                            if i[0] == 'u' and j[1][0] == 'i':
                                two_hop_agg[j[1]].append([self.time_matrix[int(i[1:])][int(j[1][1:])], i])
                            if i[0] == 'i' and j[1][0] == 'u':
                                two_hop_agg[j[1]].append([self.time_matrix[int(j[1][1:])][int(i[1:])], i])
                    for neighbor in two_hop_agg:  # neighbor node
                        neighbor_node = int(neighbor[1:])
                        p = softmax(cur_event.t, two_hop_weight)
                        influence = 0
                        if neighbor[0] == 'u':
                            neibor_embedding = self.user_lookup_embed[neighbor_node]
                            neibor_embedding = self.V_2_1(neibor_embedding)
                        if neighbor[0] == 'i':
                            neibor_embedding = self.item_lookup_embed[neighbor_node]
                            neibor_embedding = self.W_2_1(neibor_embedding)
                        for sources in two_hop_agg[neighbor]:  # multiple source nodes
                            source_node = int(sources[1][1:])
                            if sources[1][0] == 'u':
                                inter_sim = neibor_embedding * self.user_lookup_embed[source_node]
                                inter_sim = self.W_2_2(inter_sim)
                            if sources[1][0] == 'i':
                                inter_sim = neibor_embedding * self.item_lookup_embed[source_node]
                                inter_sim = self.V_2_2(inter_sim)
                            influence = influence + p[neighbor][sources[1]] * inter_sim

                        if neighbor[0] == 'u': self.user_lookup_embed[neighbor_node] = nn.Sigmoid()(neibor_embedding+influence) # 激活函数
                        if neighbor[0] == 'i': self.item_lookup_embed[neighbor_node] = nn.Sigmoid()(neibor_embedding+influence) # 激活函数

#####################################################################
                if cmd_args.hop>=3:
                    depth = 3
                    for i in two_hop:
                        three_hop.extend(get_neigbors(kg, i, 1)[1])
                    three_hop = list(set(three_hop))  # 删除相同元素
                    three_hop = del_list_element(three_hop, interaction_set)
                    for i in two_hop:
                        three_hop_weight[i] = []
                        for j in three_hop:
                            if i[0] == 'u' and j[0] == 'i':
                                if self.time_matrix[int(i[1:])][int(j[1:])] != -1:
                                    three_hop_weight[i].append([self.time_matrix[int(i[1:])][int(j[1:])], j])
                            if i[0] == 'i' and j[0] == 'u':
                                if self.time_matrix[int(j[1:])][int(i[1:])] != -1:
                                    three_hop_weight[i].append([self.time_matrix[int(j[1:])][int(i[1:])], j])
                        three_hop_weight[i].sort(reverse=True)
                        neighbor_num = math.floor(len(three_hop_weight[i]) * (1 - self.neighbor_ration))
                        three_hop_weight[i] = three_hop_weight[i][:neighbor_num]
                        for n in three_hop_weight[i]:
                            if n[1] not in three_hop_reduce:
                                three_hop_reduce.append(n[1])
                        if three_hop_weight[i] == []:
                            three_hop_weight.pop(i)
                    three_hop = three_hop_reduce
                    for i in three_hop_weight:
                        for j in three_hop_weight[i]:
                            if j[1] not in three_hop_agg:
                                three_hop_agg[j[1]] = []
                            if i[0] == 'u' and j[1][0] == 'i':
                                three_hop_agg[j[1]].append([self.time_matrix[int(i[1:])][int(j[1][1:])], i])
                            if i[0] == 'i' and j[1][0] == 'u':
                                three_hop_agg[j[1]].append([self.time_matrix[int(j[1][1:])][int(i[1:])], i])
                    for neighbor in three_hop_agg:  # neighbor node
                        neighbor_node = int(neighbor[1:])
                        p = softmax(cur_event.t, three_hop_weight)
                        influence = 0
                        if neighbor[0] == 'u':
                            neibor_embedding = self.user_lookup_embed[neighbor_node]
                            neibor_embedding = self.V_3_1(neibor_embedding)
                        if neighbor[0] == 'i':
                            neibor_embedding = self.item_lookup_embed[neighbor_node]
                            neibor_embedding = self.W_3_1(neibor_embedding)
                        for sources in three_hop_agg[neighbor]:  # multiple source nodes
                            source_node = int(sources[1][1:])
                            if sources[1][0] == 'u':
                                inter_sim = neibor_embedding * self.user_lookup_embed[source_node]
                                inter_sim = self.W_3_2(inter_sim)
                            if sources[1][0] == 'i':
                                inter_sim = neibor_embedding * self.item_lookup_embed[source_node]
                                inter_sim = self.V_3_2(inter_sim)
                            influence = influence + p[neighbor][sources[1]] * inter_sim


                        if neighbor[0] == 'u': self.user_lookup_embed[neighbor_node] = nn.Sigmoid()(neibor_embedding+influence) # 激活函数
                        if neighbor[0] == 'i': self.item_lookup_embed[neighbor_node] = nn.Sigmoid()(neibor_embedding+influence) # 激活函数


# # ######################################################################################
                    if cmd_args.hop == 4:
                        depth = 4
                        for i in three_hop:
                            four_hop.extend(get_neigbors(kg, i, 1)[1])
                        four_hop = list(set(four_hop))  # 删除相同元素
                        four_hop = del_list_element(four_hop, interaction_set)
                        four_hop = del_list_element(four_hop, one_hop)
                        four_hop = del_list_element(four_hop, two_hop)
                        four_hop = del_list_element(four_hop, three_hop)
                        for i in one_hop:
                            four_hop_weight[i] = []
                            for j in four_hop:
                                if i[0] == 'u' and j[0] == 'i':
                                    if self.time_matrix[int(i[1:])][int(j[1:])] != -1:
                                        four_hop_weight[i].append([self.time_matrix[int(i[1:])][int(j[1:])], j])
                                if i[0] == 'i' and j[0] == 'u':
                                    if self.time_matrix[int(j[1:])][int(i[1:])] != -1:
                                        four_hop_weight[i].append([self.time_matrix[int(j[1:])][int(i[1:])], j])
                            four_hop_weight[i].sort(reverse=True)
                            neighbor_num = math.floor(len(four_hop_weight[i]) * (1 - self.neighbor_ration))
                            four_hop_weight[i] = four_hop_weight[i][:neighbor_num]
                            for n in four_hop_weight[i]:
                                if n[1] not in four_hop_reduce:
                                    four_hop_reduce.append(n[1])
                            if four_hop_weight[i] == []:
                                four_hop_weight.pop(i)
                        four_hop = four_hop_reduce
                        for i in four_hop_weight:
                            for j in four_hop_weight[i]:
                                if j[1] not in four_hop_agg:
                                    four_hop_agg[j[1]] = []
                                if i[0] == 'u' and j[1][0] == 'i':
                                    four_hop_agg[j[1]].append([self.time_matrix[int(i[1:])][int(j[1][1:])], i])
                                if i[0] == 'i' and j[1][0] == 'u':
                                    four_hop_agg[j[1]].append([self.time_matrix[int(j[1][1:])][int(i[1:])], i])
                        for neighbor in four_hop_agg:  # neighbor node
                            neighbor_node = int(neighbor[1:])
                            p = softmax(cur_event.t, four_hop_weight)
                            influence = 0
                            if neighbor[0] == 'u':
                                neibor_embedding = self.user_lookup_embed[neighbor_node]
                                neibor_embedding = self.V_4_1(neibor_embedding)
                            if neighbor[0] == 'i':
                                neibor_embedding = self.item_lookup_embed[neighbor_node]
                                neibor_embedding = self.W_4_1(neibor_embedding)
                            for sources in four_hop_agg[neighbor]:  # multiple source nodes
                                source_node = int(sources[1][1:])
                                if sources[1][0] == 'u':
                                    inter_sim = neibor_embedding * self.user_lookup_embed[source_node]
                                    inter_sim = self.W_4_2(inter_sim)
                                if sources[1][0] == 'i':
                                    inter_sim = neibor_embedding * self.item_lookup_embed[source_node]
                                    inter_sim = self.V_4_2(inter_sim)
                                influence = influence + p[neighbor][sources[1]] * inter_sim

                            if neighbor[0] == 'u': self.user_lookup_embed[neighbor_node] = nn.Sigmoid()(
                                neibor_embedding + influence)  # 激活函数
                            if neighbor[0] == 'i': self.item_lookup_embed[neighbor_node] = nn.Sigmoid()(
                                neibor_embedding + influence)  # 激活函数



                if phase == 'test':  # update embeddings into the embed mat
                    self.updated_user_embed[cur_event.user] = self.user_lookup_embed[cur_event.user]
                    self.updated_item_embed[cur_event.item] = self.item_lookup_embed[cur_event.item]
                    for i in user_set:
                        i=int(i[1:])
                        self.updated_user_embed[i] = self.user_lookup_embed[i]
                    for j in item_set:
                        j = int(j[1:])
                        self.updated_item_embed[j] = self.item_lookup_embed[j]
                cur_time.update_event(cur_event.user, cur_event.item, cur_event.t)

            rmse = torch.sqrt(mse / len(events)).item()
            mae = mae.item() / len(events)
            torch.set_grad_enabled(True)
            if phase == 'train':
                return loss, mae, rmse
            else:
                gt = self.user_ground
                HR_5 = computeHR(user_rec_5, gt)
                HR_10 = computeHR(user_rec_10, gt)
                HR_20 = computeHR(user_rec_20, gt)
                return loss / len(events), HR_5, HR_10, HR_20, mae, rmse




def del_list_element(list,element):
    diff_list=[]
    for item in list:
        if item not in element:
            diff_list.append(item)
    return diff_list


def user_topK(k, user_rec, cur_event, ranks):
    user = cur_event.user
    if len(user_rec[user]) >= k:
        user_rec[user].clear()
    for item_id, rank in enumerate(ranks):
        if rank < (k + 1):
            user_rec[user].append(item_id)
    return user_rec


def computeHR(user_rec, gt):
    hit = 0
    GT = 0
    for user in user_rec:
        GT += len(gt[user])
        for item in user_rec[user]:
            if item in gt[user]:
                hit += 1
    HR = hit / GT
    return HR

