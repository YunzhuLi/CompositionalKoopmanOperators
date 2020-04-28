#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py \
	--env Swim \
	--len_seq 64 \
	--I_factor 10 \
	--batch_size 8 \
	--lr 1e-4 \
	--g_dim 32 \
	--pstep 2 \
	--fit_type structured \
	--log_per_iter 100 \
	--regular_data 1 \
	--gen_data 1 \
