from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from coevolve.model.utils import time_trunc
from coevolve.common.recorder import cur_time
from coevolve.common.cmd_args import cmd_args
from coevolve.common.consts import DEVICE, t_float


class ReyleighProc(object):
    
    @staticmethod
    def base_compatibility(embed_user, embed_item):
        inn = torch.sum(embed_item * embed_user, dim=-1)
        if cmd_args.int_act == 'softplus':
            return F.softplus(inn)
        else:
            assert cmd_args.int_act == 'exp'
            return torch.exp(inn)

    @staticmethod
    def time_mean(base_comp):
        t = 1.0 / torch.sqrt(base_comp)
        return t * np.sqrt(np.pi / 2)

    @staticmethod
    def survival(cur_user, cur_uembed, cur_item, cur_iembed,
                 user_embeds, user_ids, item_embeds, item_ids, t_end):        
        u_dur = [time_trunc(t_end - cur_time.get_cur_time(u, cur_item)) for u in user_ids]
        u_dur = torch.tensor(u_dur, dtype=t_float).to(DEVICE)
        item_centric = ReyleighProc.base_compatibility(user_embeds, cur_iembed) * (u_dur ** 2) * 0.5

        i_dur = [time_trunc(t_end - cur_time.get_cur_time(cur_user, i)) for i in item_ids]
        i_dur = torch.tensor(i_dur, dtype=t_float).to(DEVICE)
        user_centric = ReyleighProc.base_compatibility(cur_uembed, item_embeds) * (i_dur ** 2) * 0.5

        return torch.sum(user_centric) + torch.sum(item_centric)
        #取item——centric列中的最大
