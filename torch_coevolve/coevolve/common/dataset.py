from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import networkx as nx
from coevolve.common.cmd_args import cmd_args
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Event(object):
    def __init__(self, user, item, t, phase):
        self.user = user
        self.item = item
        self.t = t
        self.phase = phase

        self.next_user_event = None
        self.prev_user_event = None
        self.prev_item_event = None
        self.global_idx = None


class Dataset(object):
    def __init__(self):
        self.user_event_lists = []
        self.item_event_lists = []
        self.ordered_events = []
        self.num_events = 0
        self.user_list = []
        self.item_list = []
    def load_events(self, ratio,filename, phase):
        self.user_event_lists = [[] for _ in range(cmd_args.num_users)]
        self.item_event_lists = [[] for _ in range(cmd_args.num_items)]

        count = -1
        for count, line in enumerate(open(filename, 'rU')):
            pass
        count += 1
        count=count*ratio
        with open(filename, 'r') as f:
            rows = f.readlines()
            for row in rows:
                user, item, t = row.split()[:3]
                user = int(user)
                item = int(item)
                t = float(t) * cmd_args.time_scale
                cur_event = Event(user, item, t, phase)
                self.ordered_events.append(cur_event)
                count=count-1
                if count<=-1:break
        
        self.ordered_events = sorted(self.ordered_events, key=lambda x: x.t)
        for i in range(len(self.ordered_events)):
            cur_event = self.ordered_events[i]

            cur_event.global_idx = i
            user = cur_event.user
            item = cur_event.item
            if user not in self.user_list:
                self.user_list.append(user)
            if item not in self.item_list:
                self.item_list.append(item)

            if len(self.user_event_lists[user]):
                cur_event.prev_user_event = self.user_event_lists[user][-1]
            if len(self.item_event_lists[item]):
                cur_event.prev_item_event = self.item_event_lists[item][-1]
            if cur_event.prev_user_event is not None:
                cur_event.prev_user_event.next_user_event = cur_event
            self.user_event_lists[user].append(cur_event)
            self.item_event_lists[item].append(cur_event)

        self.num_events = len(self.ordered_events)

    def clear(self):
        self.user_event_lists = []
        self.item_event_lists = []
        self.ordered_events = []

train_data = Dataset()
test_data = Dataset()



def create_kg(kg,time_matrix,T_begain,user_event_list,item_event_list):

    for i in user_event_list:
        for j in i:
            ur='u'+str(j.user)
            im='i'+str(j.item)
            if j.t<=T_begain:
                kg.add_node(ur)
                kg.add_node(im)
                kg.add_edge(ur,im,time=j.t)
                time_matrix[j.user][j.item] = j.t
                # time_matrix[j.item][j.user] = j.t
    # for i in item_event_list:
    #     for j in i:
    #         ur = 'u' + str(j.user)
    #         im = 'i' + str(j.item)
    #         if j.t <= T_begain:
    #             kg.add_node(ur)
    #             kg.add_node(im)
    #             kg.add_edge(ur, im, time=j.t)
    return kg

def kg_add(kg,time_matrix,node1,node2,t):
    kg.add_node(node1)
    kg.add_node(node2)
    kg.add_edge(node1,node2,time=t)
    time_matrix[int(node1[1:])][int(node2[1:])]=t


def create_list(filename):
    user_list=[]
    item_list=[]
    with open(filename, 'r') as f:
        rows = f.readlines()
        for row in rows:
            user, item, t = row.split()[:3]
            user = int(user)
            item = int(item)
            if user not in user_list:
                user_list.append(user)
            if item not in item_list:
                item_list.append(item)
    return user_list,item_list

def merge_list(list1,list2):
    list=[]
    for i in list1 :
        if i not in list:
            list.append(i)
    for j in list2:
        if j not in list:
            list.append(j)
    list=sorted(list)
    return list

def get_neigbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output

# def get_reduce_neighbors()

def kg_test():
    kg=nx.Graph()
    kg_add(kg,1,2,5)
    kg_add(kg, 3, 4, 20)
    kg_add(kg, 1, 5, 10)
    kg_add(kg, 3, 2, 25)
    threshold(kg,10)


def threshold(kg,time):
    for edge in kg.edges():
        remove=kg.get_edge_data(edge[0],edge[1])['time']
        if remove<time:
            kg.remove_edge(edge[0],edge[1])
    return kg

def get_edge_time(kg,h,t):
    time=kg.get_edge_data(h,t)['time']
    return time

def softmax(current_time,time_list):
    weight_agg={}
    weight = {}
    for source in time_list:
        weight[source]=[]
        exp_list = []
        sum_exp_x = 0
        for x in time_list[source]:
            x=x[0]-current_time
            exp_x = np.exp(x)
            if exp_x==0:
                exp_x=0.00001
            exp_list.append(exp_x)
            sum_exp_x = sum_exp_x+exp_x
        for id,x in enumerate(time_list[source]):
            weight[source].append([x[1],exp_list[id]/sum_exp_x])

    for i in weight:
        for j in weight[i]:
            if j[0] not in weight_agg:
                weight_agg[j[0]]={}
            if i not in weight_agg[j[0]]:
                weight_agg[j[0]][i]=j[1]
    return weight_agg


def removeEmptyDict(data):
    data2 = {}
    for o in data:
        if not data[o] ==[]:
            data2[o] = data[o]
    return data2