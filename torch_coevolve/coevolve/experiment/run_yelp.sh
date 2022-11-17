#!/bin/bash

#dropbox=../../dropbox

db_name=yelp
pct=0.7
dataset=${db_name}_${pct}
DATA_ROOT=./full_data/$dataset
RESULT_ROOT=$HOME/scratch/results/torch_coevolve/$dataset

n_embed=256
learning_rate=0.001
time_scale=0.001
score_func=comp
dt=cur
save_dir=$RESULT_ROOT/e-${n_embed}-s-${score_func}-d-${dt}

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
    -dt_type $dt \
    -score_func $score_func \
    -save_dir $save_dir \
    -embed_dim $n_embed \
    -time_lb 0.1 \
    -time_ub 0.1 \
    -iters_per_val 500 \
    -max_norm -1 \
    -int_act softplus \
    "$@"

sleep  500000