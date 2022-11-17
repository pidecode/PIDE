from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from coevolve.common.cmd_args import cmd_args

def time_trunc(t):
    if t < cmd_args.time_lb:
        t = cmd_args.time_lb
    if t > cmd_args.time_ub:
        t = cmd_args.time_ub
    return t
