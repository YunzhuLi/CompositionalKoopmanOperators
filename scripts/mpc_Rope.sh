#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python shoot.py \
  --env Rope \
  --pstep 2 \
  --g_dim 32 \
  --len_seq 64 \
  --I_factor 10 \
  --fit_type structured \
  --optim_type qp \
  --fit_num 8 \
  --roll_step 40 \
  --roll_start 0 \
  --feedback 40 \
  --shoot_set demo \
