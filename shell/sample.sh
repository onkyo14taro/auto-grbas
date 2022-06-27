#!/bin/bash

# Make RNN deterministic.
# See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
export CUBLAS_WORKSPACE_CONFIG=":16:8"

dir_project="$(dirname $(cd $(dirname $0); pwd))"
cd "${dir_project}/src"

python main.py \
    --exp_name DUMMY \
    --gpus 1 \
    --max_epochs 1000 \
    --earlystopping_patience 50 \
    --batch_size 2 \
    --grbas_item G \
    --fold 1 \
    --rnn_bidirectional \
    --frontend_features power ist_frq grp_dly \
    --random_speed --random_crop --frontend_freq_mask
