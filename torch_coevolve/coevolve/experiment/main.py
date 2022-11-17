from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import networkx as nx
import sys
import gc
import torch
import numpy as np
import random

import torch.optim as optim

current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + "../")
sys.path.append(root_path)

from coevolve.common.cmd_args import cmd_args
from coevolve.common.consts import DEVICE
from coevolve.common.dataset import *
from coevolve.common.bipartite_graph import bg
from coevolve.common.recorder import cur_time, dur_dist
from coevolve.model.deepcoevolve import DeepCoevolve
from tqdm import tqdm


def load_data():
    train_data.load_events(1,cmd_args.train_file, 'train')
    test_data.load_events(1,cmd_args.test_file, 'test')

    for e_idx, cur_event in enumerate(test_data.ordered_events):
        cur_event.global_idx += train_data.num_events
        if cur_event.prev_user_event is None:
            continue
        train_events = train_data.user_event_lists[cur_event.user]
        if len(train_events) == 0:
            continue
        assert train_events[-1].t <= cur_event.t
        cur_event.prev_user_event = train_events[-1]
        cur_event.prev_user_event.next_user_event = cur_event

    print('# train:', train_data.num_events, '# test:', test_data.num_events)
    print('totally', cmd_args.num_users, 'users,', cmd_args.num_items, 'items')


def main_loop():
    sum_HR_5=0
    result=cmd_args.save_dir+'result_%s.txt'%(cmd_args.dataset)
    f = open(result, 'w')
    bg.reset()
    for event in train_data.ordered_events:
        bg.add_event(event.user, event.item)

    model = DeepCoevolve(train_data,test_data,num_users=cmd_args.num_users,num_items=cmd_args.num_items,embed_size=cmd_args.embed_dim,
                         score_func=cmd_args.score_func, dt_type=cmd_args.dt_type, max_norm=cmd_args.max_norm).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate)

    cur_time.reset(0)
    for e in train_data.ordered_events:
        cur_time.update_event(e.user, e.item, e.t)    
    rc_dump = cur_time.dump()

    for epoch in range(cmd_args.num_epochs):
        torch.cuda.empty_cache()
        cur_time.load_dump(*rc_dump)
        mar, HR_5, HR_10, HR_20, mae, rmse = model(train_data.ordered_events[-1].t,
                                                   test_data.ordered_events[:1000],  # 前1000个数据
                                                   phase='test')

        print('MAR:', mar, 'HR@5:', HR_5, 'HR@10:', HR_10, 'HR@20:', HR_20)
        sum_HR_5 += HR_5
        avg_HR_5 = sum_HR_5 / (epoch + 1)
        f = open(result, 'a')
        f.write(
            'MAR:  ' + str(mar) + ' ' + 'HR@5:  ' + str(HR_5) + ' ' + 'HR@10:  ' + str(HR_10) + ' ' + 'HR@20:  ' + str(
                HR_20))
        f.write('\n')
        f.close()
        pbar = tqdm(range(cmd_args.iters_per_val))


        for it in pbar:
            cur_pos = np.random.randint(train_data.num_events - cmd_args.bptt)

            T_begin = 0
            if cur_pos:
                T_begin = train_data.ordered_events[cur_pos - 1].t
            
            event_mini_batch = train_data.ordered_events[cur_pos:cur_pos + cmd_args.bptt]
            
            optimizer.zero_grad()
            loss, mae, rmse = model(T_begin,event_mini_batch,phase='train')
            pbar.set_description('epoch: %.2f, loss: %.4f, mae: %.4f, rmse: %.4f' % (epoch + (it + 1) / len(pbar), loss.item(), mae, rmse))
            
            loss.backward()
            del loss,mae,rmse
            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)

            optimizer.step()
            model.normalize()
        dur_dist.print_dist()


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    # kg_test()

    load_data()
    torch.backends.cudnn.benchmark = True
    # gc.enable()
    main_loop()
