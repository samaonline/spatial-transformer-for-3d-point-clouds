#!/bin/bash

# enter environment if using conda
# source activate caffe

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z $EXP_DIR ]; then EXP_DIR=$SCRIPT_DIR; fi
if [ -z $SPLT_CODE ]; then SPLT_CODE="$SCRIPT_DIR/../../splatnet"; fi
if [ -z $SPLT_DATA ]; then SPLT_DATA="$SCRIPT_DIR/../../data"; fi
if [ -z $SKIP_TRAIN ]; then SKIP_TRAIN=0; fi
if [ -z $SKIP_TEST ]; then SKIP_TEST=1; fi
if [ -z "$CATS" ]; then CATS="6"; fi

mkdir -p $EXP_DIR

# train
if [ $SKIP_TRAIN -le 0 ]; then
for CAT in $CATS; do
mkdir -p snaps
NETWORK_PATH="base_net2_sum.prototxt"
srun --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=base_net_lr1e-5 --kill-on-bad-exit=1 --partition Image python $SPLT_CODE/partseg3d/train.py $EXP_DIR/snaps \
    --categories $CAT \
    --feat x_y_z \
    --gpus 2 \
    --init_model snaps/6_base_net2_lr1e-5_iter_2600.caffemodel \
    --network $NETWORK_PATH \
    --prefix base_net2_sum_lr1e-5 \
    --base_lr 1e-5 --lr_decay 0.2 --stepsize 2000 --num_iter 20000 --test_interval 50 --snapshot_interval 50 --iter_size 10 \
    2>&1 | tee $EXP_DIR/snaps/base_net2_sum_lr1e-5.log; done ;
fi

# test & plot
if [ $SKIP_TEST -le 0 ]; then
srun --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=bilateral --kill-on-bad-exit=1 --partition Test python $SPLT_CODE/partseg3d/test.py shapenetB \
    --dataset_params root $SPLT_DATA/shapenetB_ericyi_ply \
    --categories $CATS \
    --snapshot best_loss \
    --sample_size -1 --batch_size 1 \
    --exp_dir $CAT/$EXP_DIR ;
fi
