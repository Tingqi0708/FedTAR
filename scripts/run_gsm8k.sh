#!/usr/bin/env bash
set -e

python train_fedtar.py \
  --model_name Qwen/Qwen1.5-MoE-A2.7B \
  --domain gsm8k \
  --train_domain auto \
  --data_dir data \
  --anchor_file data/anchors.jsonl \
  --n_anchors 80 \
  --anchor_groups 3 \
  --seed 0 \
  --split_tag dirichlet_hash_alpha0.1 \
  --clients 10 \
  --rounds 100 \
  --local_steps 5 \
  --local_batch_size 16 \
  --optimizer sgd \
  --lr 1e-5 \
  --max_len 256 \
  --min_ans_tokens 8 \
  --grad_clip 2.0 \
  --dtype bf16 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --task_dim 8 \
  --num_clusters 3 \
  --beta_router 0.5 \
  --eval_every 10 \
  --eval_n 200 \
  --eval_split test \
  --metrics_file logs/fedtar_gsm8k_alpha0.1.jsonl \
  --wandb_project fedtar \
  --wandb_run_name fedtar_gsm8k_alpha0.1 \
  --wandb_mode online
