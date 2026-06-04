# FedTAR: Task-Aware Federated Fine-Tuning for MoE-based LLMs

This repository provides the implementation of **FedTAR**, a task-aware federated fine-tuning framework for MoE-based large language models.

FedTAR focuses on federated LoRA fine-tuning under heterogeneous client distributions. The key design is to keep the LoRA `A` matrices fixed across clients, while only training, uploading, and aggregating the LoRA `B` matrices. To better handle task heterogeneity, FedTAR extracts MoE routing features from each client and reconstructs task-aware updates through a learned task subspace.

The main training script is:

```bash
fedtar_bonly_fixedA.py
```

---

## 1. Overview

FedTAR is designed for federated fine-tuning of sparse MoE-based LLMs.

In each communication round:

1. The server sends the current global LoRA state to each client.
2. Each client keeps LoRA `A` fixed and locally trains only LoRA `B`.
3. Each client computes a routing signature from MoE router outputs using anchor prompts.
4. Routing signatures are projected into a low-dimensional task space.
5. Clients are clustered based on task representations.
6. Client updates are aggregated with task-aware intra-cluster and inter-cluster weights.
7. A task subspace is fitted between routing coordinates and LoRA-B updates.
8. The router-aware update is reconstructed from the task subspace.
9. The final global update combines FedAvg and the reconstructed task-aware update.

The final update is:

```text
Delta_global = (1 - beta) * Delta_FedAvg + beta * Delta_TAR
```

where:

- `Delta_FedAvg` is the standard client-size-weighted FedAvg update.
- `Delta_TAR` is the FedTAR task-aware reconstructed update.
- `beta` is controlled by `--beta_router`.

---

## 2. Method Components

### Fixed-A B-only LoRA

FedTAR fixes LoRA `A` matrices and only trains LoRA `B` matrices.

This means:

- LoRA `A` is shared and fixed across all clients.
- LoRA `B` is locally updated.
- Only LoRA `B` updates are uploaded and aggregated.

This reduces communication cost and avoids directly averaging both low-rank factors.

---

### Router-based Task Representation

For each client, FedTAR extracts a routing signature from MoE router outputs on anchor prompts.

The routing signature is:

```text
x_i = [mu_i ; vec(C_i)]
```

where:

- `mu_i` is the mean expert routing distribution.
- `C_i` is the top-2 expert co-activation matrix.
- `vec(C_i)` is the flattened co-activation matrix.

This routing signature captures the client-specific task preference over MoE experts.

---

### Task Space Projection

Client routing signatures are projected into a low-dimensional task space using SVD/PCA.

The task dimension is controlled by:

```bash
--task_dim
```

Example:

```bash
--task_dim 8
```

---

### Clustering-based Aggregation

FedTAR clusters clients in the task space using k-means.

The number of clusters is controlled by:

```bash
--num_clusters
```

Example:

```bash
--num_clusters 3
```

Aggregation is performed in two stages:

1. **Intra-cluster aggregation**

   Clients in the same cluster are weighted by their distance to the cluster center and their local data size.

2. **Inter-cluster aggregation**

   Cluster-level updates are further aggregated according to their distance to the global task center and total cluster data size.

---

### Task-aware Reconstruction

FedTAR learns a task subspace between client task coordinates and client LoRA-B updates:

```text
Delta ≈ UZ
```

where:

- `Delta` is the matrix of flattened client LoRA-B updates.
- `Z` is the matrix of client task coordinates.
- `U` maps task coordinates to parameter updates.

The task-aware update is reconstructed as:

```text
Delta_TAR = U z_agg
```

The final global update is:

```text
Delta_global = (1 - beta) * Delta_FedAvg + beta * Delta_TAR
```

---

## 3. Environment Setup

Install the required packages:

```bash
pip install torch transformers peft tqdm wandb
```

For dataset preparation, also install:

```bash
pip install datasets numpy
```

Recommended environment:

```bash
python >= 3.9
torch >= 2.0
transformers
peft
wandb
tqdm
datasets
numpy
```

The default model is:

```bash
Qwen/Qwen1.5-MoE-A2.7B
```

You can specify another model with:

```bash
--model_name
```

---

## 4. Datasets

We evaluate FedTAR on four benchmark tasks:

| Domain | Task Type | Metric |
|---|---|---|
| `gsm8k` | Math reasoning | Exact Match |
| `mmlu` | Knowledge understanding | Accuracy |
| `bbh` | Complex reasoning | Exact Match |
| `dolly` | Instruction following | ROUGE-L |

The training script expects all datasets to be converted into JSONL format.

Each example should contain at least:

```json
{
  "prompt": "The input instruction or question.",
  "answer": "The target response or final answer."
}
```

---

## 5. Data Preparation

This repository assumes that all datasets have already been downloaded, formatted, and split into federated clients.

The training script does **not** automatically download or preprocess datasets.

### 5.1 Training Data

FedTAR follows the experimental setup below:

- For GSM8K, MMLU, and BBH, we use Alpaca-GPT-4 style instruction-tuning data for federated training.
- For Dolly-15K, we use Dolly task-category data and partition clients based on task categories.
- Evaluation is performed on the corresponding benchmark test sets.

The default training-domain behavior is:

```bash
--train_domain auto
```

Under this setting:

```text
domain = gsm8k / mmlu / bbh  ->  train_domain = alpaca_gpt4
domain = dolly              ->  train_domain = dolly
```

That means when running GSM8K, MMLU, or BBH experiments, the script reads federated training data from:

```bash
data/fl_splits/seed{seed}/{split_tag}/alpaca_gpt4/client_{cid}/train.jsonl
```

When running Dolly experiments, the script reads federated training data from:

```bash
data/fl_splits/seed{seed}/{split_tag}/dolly/client_{cid}/train.jsonl
```

---

### 5.2 Federated Client Split

The client training files should be placed under:

```bash
data/fl_splits/seed{seed}/{split_tag}/{train_domain}/client_{cid}/train.jsonl
```

For example, for GSM8K with Alpaca-GPT-4 training data:

```bash
data/fl_splits/seed0/dirichlet_hash_alpha0.1/alpaca_gpt4/client_00/train.jsonl
data/fl_splits/seed0/dirichlet_hash_alpha0.1/alpaca_gpt4/client_01/train.jsonl
data/fl_splits/seed0/dirichlet_hash_alpha0.1/alpaca_gpt4/client_02/train.jsonl
...
data/fl_splits/seed0/dirichlet_hash_alpha0.1/alpaca_gpt4/client_09/train.jsonl
```

For Dolly:

```bash
data/fl_splits/seed0/dirichlet_hash_alpha0.1/dolly/client_00/train.jsonl
data/fl_splits/seed0/dirichlet_hash_alpha0.1/dolly/client_01/train.jsonl
data/fl_splits/seed0/dirichlet_hash_alpha0.1/dolly/client_02/train.jsonl
...
data/fl_splits/seed0/dirichlet_hash_alpha0.1/dolly/client_09/train.jsonl
```

A typical non-IID split can be created as follows:

1. Convert the original training data into JSONL format.
2. Shuffle the training set using a fixed random seed.
3. Partition the data across clients with a Dirichlet distribution.
4. Save each client partition as `client_xx/train.jsonl`.

The concentration parameter `alpha` controls heterogeneity:

- Smaller `alpha`, such as `0.1`, creates stronger non-IID heterogeneity.
- Larger `alpha`, such as `0.9`, creates milder heterogeneity.
- Very large `alpha` approaches an IID split.

We use split tags such as:

```bash
dirichlet_hash_alpha0.1
dirichlet_hash_alpha0.5
dirichlet_hash_alpha0.9
```

---

### 5.3 Evaluation Data

Evaluation files should be shared across all methods and placed under:

```bash
data/fixed_eval/seed{seed}/
```

Example:

```bash
data/fixed_eval/seed0/gsm8k_val.jsonl
data/fixed_eval/seed0/gsm8k_test.jsonl
data/fixed_eval/seed0/mmlu_val.jsonl
data/fixed_eval/seed0/mmlu_test.jsonl
data/fixed_eval/seed0/bbh_val.jsonl
data/fixed_eval/seed0/bbh_test.jsonl
data/fixed_eval/seed0/dolly_val.jsonl
data/fixed_eval/seed0/dolly_test.jsonl
```

When running:

```bash
--domain gsm8k --eval_split test --data_dir data --seed 0
```

the script evaluates on:

```bash
data/fixed_eval/seed0/gsm8k_test.jsonl
```

---

### 5.4 Anchor Prompts

FedTAR requires anchor prompts to extract router signatures.

The anchor file should be:

```bash
data/anchors.jsonl
```

Each line should contain:

```json
{
  "prompt": "The anchor prompt text."
}
```

Anchor prompts are used only for router probing.  
They are **not** used as supervised training data.

A simple way to construct anchors:

1. Collect prompts from the training data.
2. Shuffle them with the same random seed.
3. Select a fixed number of prompts, such as 80 or 200.
4. Save them into `data/anchors.jsonl`.

The number of anchors is controlled by:

```bash
--n_anchors
```

The number of anchor groups is controlled by:

```bash
--anchor_groups
```

Example:

```bash
--n_anchors 80 --anchor_groups 3
```

---

## 6. Expected Data Directory

After data preparation, the directory should look like:

```bash
data/
├── anchors.jsonl
├── fixed_eval/
│   └── seed0/
│       ├── gsm8k_val.jsonl
│       ├── gsm8k_test.jsonl
│       ├── mmlu_val.jsonl
│       ├── mmlu_test.jsonl
│       ├── bbh_val.jsonl
│       ├── bbh_test.jsonl
│       ├── dolly_val.jsonl
│       └── dolly_test.jsonl
└── fl_splits/
    └── seed0/
        ├── dirichlet_hash_alpha0.1/
        │   ├── alpaca_gpt4/
        │   │   ├── client_00/train.jsonl
        │   │   ├── client_01/train.jsonl
        │   │   └── ...
        │   └── dolly/
        │       ├── client_00/train.jsonl
        │       ├── client_01/train.jsonl
        │       └── ...
        ├── dirichlet_hash_alpha0.5/
        │   ├── alpaca_gpt4/
        │   └── dolly/
        └── dirichlet_hash_alpha0.9/
            ├── alpaca_gpt4/
            └── dolly/
```

---

## 7. Main Running Commands

### 7.1 GSM8K

```bash
python fedtar_bonly_fixedA.py \
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
```

---

### 7.2 MMLU

```bash
python fedtar_bonly_fixedA.py \
  --model_name Qwen/Qwen1.5-MoE-A2.7B \
  --domain mmlu \
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
  --min_ans_tokens 1 \
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
  --metrics_file logs/fedtar_mmlu_alpha0.1.jsonl \
  --wandb_project fedtar \
  --wandb_run_name fedtar_mmlu_alpha0.1 \
  --wandb_mode online
```

---

### 7.3 BBH

```bash
python fedtar_bonly_fixedA.py \
  --model_name Qwen/Qwen1.5-MoE-A2.7B \
  --domain bbh \
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
  --min_ans_tokens 1 \
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
  --metrics_file logs/fedtar_bbh_alpha0.1.jsonl \
  --wandb_project fedtar \
  --wandb_run_name fedtar_bbh_alpha0.1 \
  --wandb_mode online
```

---

### 7.4 Dolly-15K

```bash
python fedtar_bonly_fixedA.py \
  --model_name Qwen/Qwen1.5-MoE-A2.7B \
  --domain dolly \
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
  --metrics_file logs/fedtar_dolly_alpha0.1.jsonl \
  --wandb_project fedtar \
  --wandb_run_name fedtar_dolly_alpha0.1 \
  --wandb_mode online
```

---

## 8. Running Different Non-IID Settings

To run different heterogeneity levels, change `--split_tag`.

### Alpha = 0.1

```bash
--split_tag dirichlet_hash_alpha0.1
```

### Alpha = 0.5

```bash
--split_tag dirichlet_hash_alpha0.5
```

### Alpha = 0.9

```bash
--split_tag dirichlet_hash_alpha0.9
```

Example:

```bash
python fedtar_bonly_fixedA.py \
  --domain gsm8k \
  --train_domain auto \
  --data_dir data \
  --anchor_file data/anchors.jsonl \
  --split_tag dirichlet_hash_alpha0.5 \
  --clients 10 \
  --rounds 100 \
  --local_steps 5 \
  --local_batch_size 16 \
  --optimizer sgd \
  --lr 1e-5 \
  --task_dim 8 \
  --num_clusters 3 \
  --beta_router 0.5 \
  --metrics_file logs/fedtar_gsm8k_alpha0.5.jsonl \
  --wandb_run_name fedtar_gsm8k_alpha0.5
```

---

## 9. Running Without W&B

To disable Weights & Biases logging, use:

```bash
--wandb_mode disabled
```

Example:

```bash
python fedtar_bonly_fixedA.py \
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
  --wandb_mode disabled
```

---

## 10. Important Arguments

### Model and LoRA

| Argument | Description | Default |
|---|---|---|
| `--model_name` | Hugging Face model name | `Qwen/Qwen1.5-MoE-A2.7B` |
| `--dtype` | Model dtype, `bf16` or `fp16` | `bf16` |
| `--lora_r` | LoRA rank | `8` |
| `--lora_alpha` | LoRA scaling factor | `16` |
| `--lora_dropout` | LoRA dropout rate | `0.05` |

---

### Federated Learning

| Argument | Description | Default |
|---|---|---|
| `--clients` | Number of clients | `10` |
| `--rounds` | Number of communication rounds | `100` |
| `--local_steps` | Number of local update steps per round | `5` |
| `--local_batch_size` | Local batch size | `16` |
| `--optimizer` | Local optimizer, e.g., `sgd` | `sgd` |
| `--lr` | Local learning rate | `1e-5` |
| `--max_len` | Maximum input sequence length | `256` |
| `--grad_clip` | Gradient clipping norm | `2.0` |

---

### Data

| Argument | Description | Default |
|---|---|---|
| `--domain` | Evaluation domain: `gsm8k`, `mmlu`, `bbh`, or `dolly` | required |
| `--train_domain` | Training domain. Use `auto` for default setup | `auto` |
| `--data_dir` | Root data directory | `data` |
| `--anchor_file` | Anchor prompt file | required |
| `--n_anchors` | Number of anchor prompts | `80` |
| `--anchor_groups` | Number of anchor groups | `3` |
| `--seed` | Random seed | `0` |
| `--split_tag` | Federated split tag | `dirichlet_hash_alpha0.1` |

---

### FedTAR Aggregation

| Argument | Description | Default |
|---|---|---|
| `--probe_layers` | MoE router layers used for probing | `all` |
| `--router_temp` | Router softmax temperature | `1.0` |
| `--task_dim` | Task-space dimension | `8` |
| `--ridge_lambda` | Ridge regularization coefficient | `1e-4` |
| `--num_clusters` | Number of k-means clusters | `3` |
| `--kmeans_iters` | Number of k-means iterations | `20` |
| `--tau_intra` | Intra-cluster distance temperature | `1.0` |
| `--tau_inter` | Inter-cluster distance temperature | `1.0` |
| `--beta_router` | Weight of task-aware reconstructed update | `0.5` |

---

### Evaluation and Logging

| Argument | Description | Default |
|---|---|---|
| `--eval_n` | Number of evaluation samples. `0` means all samples | `0` |
| `--eval_every` | Evaluate every N communication rounds | `10` |
| `--eval_split` | Evaluation split, `val` or `test` | `test` |
| `--metrics_file` | JSONL metrics output file | `logs/fedtar_metrics.jsonl` |
| `--wandb_project` | W&B project name | `fedtar` |
| `--wandb_run_name` | W&B run name | `fedtar_bonly_fixedA` |
| `--wandb_mode` | `online`, `offline`, or `disabled` | `online` |

---

## 11. Output Logs

FedTAR logs training, aggregation, and evaluation statistics to both W&B and a JSONL file.

Each round records:

- mean client loss;
- valid local optimization steps;
- skipped micro-batches;
- reconstruction error;
- FedAvg update norm;
- FedTAR update norm;
- final global update norm;
- cluster counts;
- inter-cluster weights;
- evaluation metric when evaluation is enabled.

Example metrics file:

```bash
logs/fedtar_gsm8k_alpha0.1.jsonl
```

Each line is a JSON object corresponding to one communication round.

---

## 12. Evaluation Metrics

| Domain | Metric |
|---|---|
| `gsm8k` | Exact Match |
| `mmlu` | Accuracy |
| `bbh` | Exact Match |
| `dolly` | ROUGE-L |

---

## 13. Notes

### Fixed-A B-only LoRA

FedTAR only trains and aggregates LoRA `B`.

LoRA `A` is fixed after initialization and shared across all clients.

---

### Router Probing

Router signatures are computed after each client completes local training.

Therefore, the routing signature reflects the locally adapted client model.

---

### Training Domains

By default, `--train_domain auto` maps:

```text
gsm8k / mmlu / bbh -> alpaca_gpt4
dolly              -> dolly
```

Use a custom training domain by explicitly setting:

```bash
--train_domain your_train_domain
```

Then the script will read:

```bash
data/fl_splits/seed{seed}/{split_tag}/your_train_domain/client_{cid}/train.jsonl
```

---

### Memory and Runtime

The script builds a local LoRA model for each client in each communication round.

For large MoE models, this can be memory- and time-consuming.

If GPU memory is limited, consider reducing:

```bash
--clients
--local_batch_size
--max_len
--n_anchors
--eval_n
```

To disable progress bars:

```bash
--disable_tqdm
```

To disable W&B:

```bash
--wandb_mode disabled
```

---

## 14. Citation

If you use this code, please cite the corresponding paper.

```bibtex
@article{fedtar,
  title={FedTAR: Task-Aware Federated Fine-Tuning for MoE-based Large Language Models},
  author={Anonymous},
  journal={Anonymous Submission},
  year={2026}
}
```
