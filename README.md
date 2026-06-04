# FedTAR Project

This is a modular project version of `fedtar_bonly_fixedA.py`.

## Structure

```text
fedtar_project/
├── train_fedtar.py
├── requirements.txt
├── README.md
├── fedtar/
│   ├── __init__.py
│   ├── runner.py
│   ├── utils.py
│   ├── lora_utils.py
│   ├── evaluation.py
│   ├── local_training.py
│   ├── routing.py
│   ├── subspace.py
│   └── aggregation.py
└── scripts/
    ├── run_gsm8k.sh
    ├── run_mmlu.sh
    ├── run_bbh.sh
    └── run_dolly.sh
```

## Install

```bash
pip install -r requirements.txt
```

## Data Layout

Training data:

```bash
data/fl_splits/seed{seed}/{split_tag}/{train_domain}/client_{cid}/train.jsonl
```

Evaluation data:

```bash
data/fixed_eval/seed{seed}/{domain}_test.jsonl
data/fixed_eval/seed{seed}/{domain}_val.jsonl
```

Anchor prompts:

```bash
data/anchors.jsonl
```

Each JSONL sample should contain:

```json
{"prompt": "...", "answer": "..."}
```

## Run

Use the main entry script:

```bash
python train_fedtar.py \
  --model_name Qwen/Qwen1.5-MoE-A2.7B \
  --domain gsm8k \
  --train_domain auto \
  --data_dir data \
  --anchor_file data/anchors.jsonl \
  --split_tag dirichlet_hash_alpha0.1 \
  --clients 10 \
  --rounds 100 \
  --local_steps 5 \
  --local_batch_size 16 \
  --optimizer sgd \
  --lr 1e-5 \
  --max_len 256 \
  --grad_clip 2.0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --task_dim 8 \
  --num_clusters 3 \
  --beta_router 0.5 \
  --eval_every 10 \
  --eval_n 200 \
  --metrics_file logs/fedtar_gsm8k_alpha0.1.jsonl
```

Or run a script:

```bash
bash scripts/run_gsm8k.sh
```

## Module Roles

- `fedtar/utils.py`: JSONL I/O, random seed, time formatting, tensor norm.
- `fedtar/lora_utils.py`: model loading, LoRA target discovery, LoRA state dict utilities, fixed-A/B-only control.
- `fedtar/evaluation.py`: task metrics for GSM8K, Dolly, MBPP, MMLU, and BBH.
- `fedtar/local_training.py`: response-only local training, local batching, clipping, optional DP-style upload perturbation.
- `fedtar/routing.py`: MoE router probing and routing signature extraction.
- `fedtar/subspace.py`: PCA/SVD projection, flatten/unflatten, task-subspace fitting and reconstruction.
- `fedtar/aggregation.py`: k-means, FedAvg, and clustered task-aware aggregation.
- `fedtar/runner.py`: full federated training loop.

## Notes

By default, `--train_domain auto` maps:

```text
gsm8k / mmlu / bbh -> alpaca_gpt4
dolly              -> dolly
```

Only LoRA `B` is locally trained, uploaded, and aggregated. LoRA `A` is fixed across clients.
