from typing import Dict, List, Tuple

import torch

from fedtar.lora_utils import clone_sd

# =========================================================
# Tokenization / local train
# =========================================================
def tokenize_with_response_loss(
    tokenizer,
    prompt: str,
    answer: str,
    max_len: int = 512,
    min_ans_tokens: int = 1,
):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    if len(answer_ids) < min_ans_tokens:
        return None

    eos_id = tokenizer.eos_token_id
    input_ids = prompt_ids + answer_ids + ([eos_id] if eos_id is not None else [])
    labels = [-100] * len(prompt_ids) + answer_ids + ([eos_id] if eos_id is not None else [])

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    n_valid = sum(1 for x in labels if x != -100)
    if n_valid == 0:
        return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_padded_batches(examples: List[dict], pad_token_id: int, device: torch.device) -> Dict[str, torch.Tensor]:
    max_len = max(int(ex["input_ids"].numel()) for ex in examples)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)

    for i, ex in enumerate(examples):
        L = int(ex["input_ids"].numel())
        input_ids[i, :L] = ex["input_ids"]
        attention_mask[i, :L] = ex["attention_mask"]
        labels[i, :L] = ex["labels"]

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def local_train_rows_safe(
    tokenizer,
    model,
    rows: List[dict],
    steps: int,
    lr: float,
    max_len: int,
    min_ans_tokens: int,
    local_batch_size: int = 16,
    optimizer_name: str = "sgd",
    grad_clip: float = 2.0,
    log_every: int = 50,
) -> Tuple[float, int, int]:
    """
    Local client training used in the experimental setup:
    vanilla SGD by default, local batch size 16, tau=steps local update steps,
    and gradient clipping with norm 2.0.
    """
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    opt_name = optimizer_name.lower().strip()
    if opt_name == "sgd":
        opt = torch.optim.SGD(trainable_params, lr=lr)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(trainable_params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    order = list(range(len(rows)))
    random.shuffle(order)
    ptr = 0

    running = 0.0
    valid_opt_steps = 0
    skipped_micro = 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for step in range(1, steps + 1):
        batch_examples = []
        attempts = 0
        max_attempts = max(local_batch_size * 4, local_batch_size)

        while len(batch_examples) < local_batch_size and attempts < max_attempts:
            attempts += 1
            if len(order) == 0:
                break
            if ptr >= len(order):
                random.shuffle(order)
                ptr = 0

            ex = rows[order[ptr]]
            ptr += 1

            batch_ex = tokenize_with_response_loss(
                tokenizer,
                ex["prompt"],
                ex["answer"],
                max_len=max_len,
                min_ans_tokens=min_ans_tokens,
            )
            if batch_ex is None:
                skipped_micro += 1
                continue
            batch_examples.append(batch_ex)

        if len(batch_examples) == 0:
            continue

        batch = collate_padded_batches(batch_examples, pad_id, model.device)
        valid_label_cnt = int((batch["labels"] != -100).sum().item())
        if valid_label_cnt == 0:
            skipped_micro += len(batch_examples)
            continue

        opt.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss

        if not torch.isfinite(loss):
            print("[WARN] non-finite loss detected, skip this batch")
            skipped_micro += len(batch_examples)
            continue

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        opt.step()

        running += float(loss.item())
        valid_opt_steps += 1

        if log_every > 0 and step % log_every == 0:
            avg = running / max(valid_opt_steps, 1)
            print(f"    step {step:04d}/{steps} | loss(avg)={avg:.4f} | valid_opt_steps={valid_opt_steps}")

    return running / max(valid_opt_steps, 1), valid_opt_steps, skipped_micro


@torch.no_grad()
def clip_state_dict_by_global_norm(sd: Dict[str, torch.Tensor], clip_norm: float) -> Dict[str, torch.Tensor]:
    if clip_norm <= 0:
        return clone_sd(sd)
    total_sq = 0.0
    for v in sd.values():
        total_sq += float(torch.sum(v.float() ** 2).item())
    total_norm = total_sq ** 0.5
    scale = min(1.0, clip_norm / (total_norm + 1e-12))
    return {k: (v.float() * scale).to(v.dtype).detach().cpu() for k, v in sd.items()}


@torch.no_grad()
def apply_dp_to_state_dict(
    sd: Dict[str, torch.Tensor],
    clip_norm: float,
    epsilon: float,
    noise_multiplier: float,
) -> Dict[str, torch.Tensor]:
    """
    Optional upload perturbation used to match the experimental DP setting.
    If epsilon <= 0 or noise_multiplier <= 0, the function only applies clipping.
    For a strict DP claim, use an external privacy accountant for the final epsilon.
    """
    clipped = clip_state_dict_by_global_norm(sd, clip_norm)
    if epsilon <= 0 or noise_multiplier <= 0 or clip_norm <= 0:
        return clipped
    sigma = noise_multiplier * clip_norm / max(float(epsilon), 1e-12)
    return {k: (v.float() + torch.randn_like(v.float()) * sigma).to(v.dtype).detach().cpu() for k, v in clipped.items()}

