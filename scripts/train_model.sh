#!/bin/bash

python src/train.py \
  --dataset_name 'zara1' \
  --delim $'\t' \
  --obs_len 8 \
  --pred_len 12 \
  --batch_size 64 \
  --num_epochs 200 \
  --learning_rate 1e-3 \
  --use_gpu 0 \
  --checkpoint_name 'do_tp_zara1'