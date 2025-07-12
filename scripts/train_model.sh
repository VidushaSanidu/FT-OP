#!/bin/bash

python src/train.py \
  --dataset_name 'eth' \
  --delim $'\t' \
  --obs_len 8 \
  --pred_len 12 \
  --batch_size 64 \
  --num_epochs 50 \
  --learning_rate 1e-3 \
  --use_gpu 0 \
  --print_every 10 \
  --checkpoint_name 'do_tp_zara1'


