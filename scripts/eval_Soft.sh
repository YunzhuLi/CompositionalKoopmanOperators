#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
	--env Soft \
	--pstep 2 \
	--g_dim 32 \
	--len_seq 64 \
	--I_factor 10 \
	--fit_type structured \
	--fit_num 8 \
	--eval_set demo \
