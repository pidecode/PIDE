#!/bin/bash

#dropbox=../../dropbox

db_name=reddit_1000_random
pct=0.7
dataset=${db_name}_${pct}
DATA_ROOT=./full_data/$dataset
RESULT_ROOT=$HOME/scratch/results/torch_coevolve/$dataset

n_embed=128
learning_rate=0.001
time_scale=0.001
save_dir=$RESULT_ROOT/e-${n_embed}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0


python main.py \
    -meta_file $DATA_ROOT/meta.txt \
    -train_file $DATA_ROOT/train.txt \
    -test_file $DATA_ROOT/test.txt \
    -time_scale $time_scale \
    -learning_rate $learning_rate \
    -save_dir $save_dir \
    -embed_dim $n_embed \
    -iters_per_val 100 \
    -max_norm 0.1 \
    -time_lb 0.01 \
    -time_ub 0.1 \
    -score_func log_ll \
    "$@"

sleep  500000