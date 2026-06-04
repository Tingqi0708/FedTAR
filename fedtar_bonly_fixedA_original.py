#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fedtar_bonly_fixedA.py

FedTAR with:
- fixed LoRA A across clients
- train / upload / aggregate only LoRA B
- routing signature x_i = [mu_i ; vec(C_i)]
- PCA -> z_i
- k-means clustering on z_i
- intra-cluster weighted aggregation
- inter-cluster weighted aggregation
- task-aware reconstruction
- final global update = (1-beta) * delta_fedavg + beta * delta_task

Supported domains:
- gsm8k / dolly / mbpp / mmlu / bbh
"""

import argparse
import json
import multiprocessing as mp
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import wandb
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================================================
# Basic utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_seconds(sec: float) -> str:
    sec = max(float(sec), 0.0)
    if sec < 60:
        return f"{sec:.0f}s"
    mins = sec / 60.0
    if mins < 60:
        return f"{mins:.1f}m"
    hours = mins / 60.0
    return f"{hours:.2f}h"


def safe_norm(x: torch.Tensor) -> float:
    return float(torch.norm(x.float()).item())


# =========================================================
# LoRA model build
# =========================================================
def guess_moe_lora_targets(model) -> List[str]:
    cand = set()

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        lname = name.lower()
        last = name.split(".")[-1]

        if any(x in lname for x in [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj"
        ]):
            cand.add(last)

        if "expert" in lname and any(x in lname for x in [
            "up_proj", "down_proj", "gate_proj", "w1", "w2", "w3"
        ]):
            cand.add(last)

        if "shared_expert" in lname and any(x in lname for x in [
            "up_proj", "down_proj", "gate_proj", "w1", "w2", "w3"
        ]):
            cand.add(last)

    if len(cand) == 0:
        cand.update(["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])

    return sorted(cand)


def build_lora_model(
    model_name: str,
    dtype: str = "bf16",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    target_modules = guess_moe_lora_targets(model)
    print("[INIT] LoRA target modules =", target_modules)

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return tokenizer, model, target_modules


def get_lora_state_dict(model, only_lora_B: bool = True) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if "lora_" not in k:
            continue
        if only_lora_B and ("lora_B" not in k):
            continue
        out[k] = v.detach().cpu().clone()
    return out


def get_lora_A_state_dict(model) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if "lora_A" in k:
            out[k] = v.detach().cpu().clone()
    return out


def set_partial_state_dict(model, part_sd: Dict[str, torch.Tensor]):
    if not part_sd:
        return
    cur = model.state_dict()
    for k, v in part_sd.items():
        if k in cur:
            cur[k].copy_(v.to(cur[k].device, dtype=cur[k].dtype))


def set_lora_state_dict(model, lora_sd: Dict[str, torch.Tensor]):
    if not lora_sd:
        return
    cur = model.state_dict()
    for k, v in lora_sd.items():
        if k in cur:
            cur[k].copy_(v.to(cur[k].device, dtype=cur[k].dtype))


def clone_sd(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def diff_sd(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = set(a.keys()) & set(b.keys())
    return {k: (a[k] - b[k]).detach().cpu() for k in keys}


def add_sd(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    keys = set(base.keys()) | set(delta.keys())
    for k in keys:
        if k in base and k in delta:
            out[k] = (base[k] + delta[k]).detach().cpu()
        elif k in base:
            out[k] = base[k].detach().cpu().clone()
        else:
            out[k] = delta[k].detach().cpu().clone()
    return out


def freeze_lora_A_unfreeze_lora_B(model):
    for n, p in model.named_parameters():
        if "lora_A" in n:
            p.requires_grad = False
        elif "lora_B" in n:
            p.requires_grad = True


# =========================================================
# Eval helpers
# =========================================================
def _normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_numeric_string(s: str) -> str:
    s = str(s).strip()
    s = s.replace(",", "")
    s = s.rstrip(".")
    if s == "":
        return ""
    try:
        x = float(s)
        if abs(x - int(x)) < 1e-9:
            return str(int(x))
        return str(x)
    except Exception:
        return s


def _extract_gsm8k_final_answer(s: str) -> str:
    s = str(s).strip()

    m = re.findall(r"####\s*([-+]?\d[\d,\.]*)", s)
    if m:
        return _normalize_numeric_string(m[-1])

    nums = re.findall(r"[-+]?\d[\d,\.]*", s)
    if nums:
        return _normalize_numeric_string(nums[-1])

    return _normalize_text(s)


def rouge_l_score(pred: str, ref: str) -> float:
    pred_toks = _normalize_text(pred).split()
    ref_toks = _normalize_text(ref).split()
    if len(pred_toks) == 0 or len(ref_toks) == 0:
        return 0.0

    dp = [[0] * (len(ref_toks) + 1) for _ in range(len(pred_toks) + 1)]
    for i in range(1, len(pred_toks) + 1):
        for j in range(1, len(ref_toks) + 1):
            if pred_toks[i - 1] == ref_toks[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    prec = lcs / max(len(pred_toks), 1)
    rec = lcs / max(len(ref_toks), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def generate_text(tokenizer, model, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> str:
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    if temperature <= 1e-8:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()


@torch.no_grad()
def gsm8k_em(tokenizer, model, rows: List[dict], n_samples: int = 0, max_new_tokens: int = 256) -> float:
    if n_samples and n_samples > 0:
        rows = rows[:n_samples]
    if len(rows) == 0:
        return 0.0

    correct = 0
    for ex in rows:
        pred = generate_text(tokenizer, model, ex["prompt"], max_new_tokens=max_new_tokens)
        pred_ans = _extract_gsm8k_final_answer(pred)
        gold_ans = _extract_gsm8k_final_answer(ex["answer"])
        correct += int(pred_ans == gold_ans)
    return correct / len(rows)


@torch.no_grad()
def text_rougeL(tokenizer, model, rows: List[dict], n_samples: int = 0, max_new_tokens: int = 128) -> float:
    if n_samples and n_samples > 0:
        rows = rows[:n_samples]
    if len(rows) == 0:
        return 0.0

    vals = []
    for ex in rows:
        pred = generate_text(tokenizer, model, ex["prompt"], max_new_tokens=max_new_tokens)
        vals.append(rouge_l_score(pred, ex["answer"]))
    return sum(vals) / len(vals)


def _extract_mmlu_letter(s: str) -> str:
    s = str(s).strip().upper()

    patterns = [
        r"ANSWER\s*[:：]?\s*([ABCD])\b",
        r"THE ANSWER IS\s*([ABCD])\b",
        r"OPTION\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)

    if len(s) > 0 and s[0] in "ABCD":
        return s[0]
    return ""


@torch.no_grad()
def mmlu_acc(tokenizer, model, rows: List[dict], n_samples: int = 0, max_new_tokens: int = 32) -> float:
    if n_samples and n_samples > 0:
        rows = rows[:n_samples]
    if len(rows) == 0:
        return 0.0

    correct = 0
    for ex in rows:
        pred = generate_text(tokenizer, model, ex["prompt"], max_new_tokens=max_new_tokens)
        p = _extract_mmlu_letter(pred)
        g = _extract_mmlu_letter(ex["answer"])
        correct += int(p == g)
    return correct / len(rows)


def _normalize_bbh_answer(s: str) -> str:
    s = _normalize_text(s)
    s = s.strip(" .,:;!?\"'`()[]{}")
    return s


@torch.no_grad()
def bbh_em(tokenizer, model, rows: List[dict], n_samples: int = 0, max_new_tokens: int = 128) -> float:
    if n_samples and n_samples > 0:
        rows = rows[:n_samples]
    if len(rows) == 0:
        return 0.0

    correct = 0
    for ex in rows:
        pred = generate_text(tokenizer, model, ex["prompt"], max_new_tokens=max_new_tokens)
        p = _normalize_bbh_answer(pred)
        g = _normalize_bbh_answer(ex["answer"])

        ok = (p == g)
        if not ok and len(g) > 0:
            ok = (g in p)

        correct += int(ok)
    return correct / len(rows)


SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "callable": callable,
    "chr": chr, "complex": complex, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "hash": hash,
    "hex": hex, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max, "min": min,
    "next": next, "ord": ord, "pow": pow, "print": print, "range": range,
    "repr": repr, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip,
}


def extract_python_code(text: str) -> str:
    fenced = re.findall(r"```python\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced[0].strip()
    fenced2 = re.findall(r"```\s*(.*?)```", text, flags=re.S)
    if fenced2:
        return fenced2[0].strip()
    return text.strip()


def _run_mbpp_case_worker(code: str, tests: List[str], queue: mp.Queue):
    try:
        import collections
        import functools
        import itertools
        import math as math_mod
        import re as re_mod

        safe_globals = {
            "__builtins__": SAFE_BUILTINS,
            "math": math_mod,
            "re": re_mod,
            "itertools": itertools,
            "functools": functools,
            "collections": collections,
        }
        safe_locals = {}

        exec(code, safe_globals, safe_locals)
        for t in tests:
            exec(t, safe_globals, safe_locals)

        queue.put((True, "ok"))
    except Exception as e:
        queue.put((False, repr(e)))


def run_mbpp_tests(code: str, tests: List[str], timeout_sec: int = 3) -> bool:
    if not code.strip() or tests is None or len(tests) == 0:
        return False

    q = mp.Queue()
    p = mp.Process(target=_run_mbpp_case_worker, args=(code, tests, q))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return False

    if q.empty():
        return False

    ok, _ = q.get()
    return bool(ok)


@torch.no_grad()
def mbpp_pass_at_1(
    tokenizer,
    model,
    rows: List[dict],
    n_samples: int = 0,
    max_new_tokens: int = 256,
    timeout_sec: int = 3,
) -> float:
    if n_samples and n_samples > 0:
        rows = rows[:n_samples]
    if len(rows) == 0:
        return 0.0

    passed = 0
    total = 0
    for ex in rows:
        tests = ex.get("tests", ex.get("test_list", None))
        if tests is None:
            continue

        pred = generate_text(tokenizer, model, ex["prompt"], max_new_tokens=max_new_tokens)
        code = extract_python_code(pred)
        ok = run_mbpp_tests(code, tests, timeout_sec=timeout_sec)
        passed += int(ok)
        total += 1

    if total == 0:
        raise RuntimeError("MBPP eval rows do not contain tests/test_list; pass@1 cannot be computed.")

    return passed / total


def eval_single_domain(
    domain: str,
    tokenizer,
    model,
    fixed_eval_dir: str | Path,
    n_samples: int = 0,
    split: str = "test",
):
    fixed_eval_dir = Path(fixed_eval_dir)

    def choose_file(test_name: str, val_name: str) -> Path:
        test_path = fixed_eval_dir / test_name
        val_path = fixed_eval_dir / val_name

        if split == "test" and test_path.exists():
            return test_path
        if split == "val" and val_path.exists():
            return val_path
        if test_path.exists():
            return test_path
        if val_path.exists():
            return val_path

        raise FileNotFoundError(
            f"Neither {test_name} nor {val_name} exists under {fixed_eval_dir}"
        )

    if domain == "gsm8k":
        rows = load_jsonl(choose_file("gsm8k_test.jsonl", "gsm8k_val.jsonl"))
        return "EM", gsm8k_em(tokenizer, model, rows, n_samples=n_samples, max_new_tokens=256)

    if domain == "dolly":
        rows = load_jsonl(choose_file("dolly_test.jsonl", "dolly_val.jsonl"))
        return "ROUGE-L", text_rougeL(tokenizer, model, rows, n_samples=n_samples, max_new_tokens=128)

    if domain == "mbpp":
        rows = load_jsonl(choose_file("mbpp_test.jsonl", "mbpp_val.jsonl"))
        return "pass@1", mbpp_pass_at_1(tokenizer, model, rows, n_samples=n_samples, max_new_tokens=256, timeout_sec=3)

    if domain == "mmlu":
        rows = load_jsonl(choose_file("mmlu_test.jsonl", "mmlu_val.jsonl"))
        return "ACC", mmlu_acc(tokenizer, model, rows, n_samples=n_samples, max_new_tokens=32)

    if domain == "bbh":
        rows = load_jsonl(choose_file("bbh_test.jsonl", "bbh_val.jsonl"))
        return "EM", bbh_em(tokenizer, model, rows, n_samples=n_samples, max_new_tokens=128)

    raise ValueError(domain)


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


# =========================================================
# Router helpers
# =========================================================
def _find_moe_gate_modules(model) -> List[torch.nn.Module]:
    base = getattr(model, "base_model", None)
    if base is not None:
        model = base

    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.base_model.model.layers,
        lambda m: m.base_model.model.model.layers,
    ]

    layers = None
    for getter in candidates:
        try:
            x = getter(model)
            _ = len(x)
            layers = x
            break
        except Exception:
            continue

    if layers is None:
        for n, mod in model.named_modules():
            if n.endswith("layers"):
                try:
                    _ = len(mod)
                    layers = mod
                    break
                except Exception:
                    pass

    if layers is None:
        raise RuntimeError("Cannot locate transformer layers in model.")

    gates = []
    for layer in layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            gates.append(layer.mlp.gate)
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
            gates.append(layer.mlp.router)
        else:
            gates.append(None)
    return gates


def get_probe_layers_from_model(model, probe_layers_arg: str) -> List[int]:
    gates = _find_moe_gate_modules(model)
    valid = [i for i, g in enumerate(gates) if g is not None]

    if probe_layers_arg.strip().lower() == "all":
        if len(valid) == 0:
            raise RuntimeError("No MoE router layers found.")
        return valid

    return [int(x.strip()) for x in probe_layers_arg.split(",") if x.strip()]


def split_anchor_groups(anchor_rows: List[dict], n_groups: int) -> List[List[dict]]:
    if n_groups <= 1 or len(anchor_rows) == 0:
        return [anchor_rows]

    rows = [(len(str(r["prompt"])), r) for r in anchor_rows]
    rows.sort(key=lambda x: x[0])

    groups = [[] for _ in range(n_groups)]
    for idx, (_l, r) in enumerate(rows):
        g = min((idx * n_groups) // max(len(rows), 1), n_groups - 1)
        groups[g].append(r)

    groups = [g for g in groups if len(g) > 0]
    return groups if len(groups) > 0 else [anchor_rows]


@torch.no_grad()
def compute_routing_signature(
    model,
    tokenizer,
    anchor_groups: List[List[dict]],
    layers_to_probe: List[int],
    max_len: int,
    num_experts: int,
    device: torch.device,
    temp: float = 1.0,
    normalize_feature: bool = True,
    show_probe_progress: bool = False,
) -> torch.Tensor:
    """
    x_i = [mu_i ; vec(C_i)]
    mu_i: mean routing distribution
    C_i : top-2 co-activation matrix
    """
    model.eval()
    gates = _find_moe_gate_modules(model)
    E = num_experts

    all_probs = []
    all_top2 = []

    group_iter = anchor_groups
    if show_probe_progress:
        group_iter = tqdm(group_iter, desc="Router groups", leave=False)

    for group_rows in group_iter:
        layer_to_logits: Dict[int, List[torch.Tensor]] = {l: [] for l in layers_to_probe}
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(_module, _inp, out):
                out0 = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(out0):
                    layer_to_logits[layer_idx].append(out0.detach())
            return hook_fn

        for l in layers_to_probe:
            if l < 0 or l >= len(gates) or gates[l] is None:
                raise ValueError(f"Layer {l} has no MoE gate/router module.")
            hooks.append(gates[l].register_forward_hook(make_hook(l)))

        prompt_iter = group_rows
        if show_probe_progress:
            prompt_iter = tqdm(prompt_iter, desc="Router probing", leave=False)

        for ex in prompt_iter:
            prompt = ex["prompt"]
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            ).to(device)
            _ = model(**enc)

        for h in hooks:
            h.remove()

        for l in layers_to_probe:
            if len(layer_to_logits[l]) == 0:
                continue

            flat_list = []
            for t in layer_to_logits[l]:
                if t.dim() == 3:
                    tt = t.reshape(-1, t.shape[-1])
                elif t.dim() == 2:
                    tt = t.reshape(-1, t.shape[-1])
                elif t.dim() == 1:
                    tt = t.unsqueeze(0)
                else:
                    continue
                if tt.shape[-1] != E:
                    continue
                flat_list.append(tt)

            if len(flat_list) == 0:
                continue

            logits2 = torch.cat(flat_list, dim=0).float()
            probs2 = F.softmax(logits2 / max(temp, 1e-6), dim=-1)
            top2_idx = torch.topk(probs2, k=min(2, E), dim=-1).indices

            all_probs.append(probs2.cpu())
            all_top2.append(top2_idx.cpu())

    if len(all_probs) == 0:
        dim = E + E * E
        return torch.zeros(dim, dtype=torch.float32)

    probs = torch.cat(all_probs, dim=0)
    top2 = torch.cat(all_top2, dim=0)
    T = probs.shape[0]

    mu = probs.mean(dim=0)

    C = torch.zeros((E, E), dtype=torch.float32)
    for t in range(T):
        idx = top2[t].tolist()
        if len(idx) == 1:
            a = idx[0]
            C[a, a] += 1.0
        else:
            a, b = idx[0], idx[1]
            C[a, b] += 1.0
            C[b, a] += 1.0

    C = C / max(float(T), 1.0)

    x = torch.cat([mu.float(), C.reshape(-1).float()], dim=0)

    if normalize_feature:
        x = x / (torch.norm(x, p=2) + 1e-12)

    return x


# =========================================================
# PCA / flatten / subspace
# =========================================================
@torch.no_grad()
def pca_project(X: torch.Tensor, proj_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N, D = X.shape
    d = min(proj_dim, N, D)
    if d <= 0:
        raise ValueError("Invalid proj_dim for PCA.")

    mu = X.mean(dim=0)
    Xc = X - mu[None, :]
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    P = Vh[:d].T.contiguous()
    Z = Xc @ P
    return Z, mu, P


def state_dict_template_and_keys(sds: List[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    keys = sorted(set().union(*[sd.keys() for sd in sds]))
    template = {}
    for k in keys:
        for sd in sds:
            if k in sd:
                template[k] = sd[k].detach().cpu().clone()
                break
    return template, keys


def flatten_state_dict(sd: Dict[str, torch.Tensor], template: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    vecs = []
    for k in keys:
        ref = template[k]
        t = sd.get(k, torch.zeros_like(ref))
        vecs.append(t.float().reshape(-1))
    return torch.cat(vecs, dim=0)


def unflatten_vector_to_state_dict(vec: torch.Tensor, template: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
    out = {}
    offset = 0
    for k in keys:
        ref = template[k]
        numel = ref.numel()
        out[k] = vec[offset: offset + numel].view_as(ref).to(ref.dtype).detach().cpu()
        offset += numel
    if offset != vec.numel():
        raise ValueError(f"Unflatten mismatch: used={offset}, total={vec.numel()}")
    return out


@torch.no_grad()
def fit_task_subspace_ridge(Delta: torch.Tensor, Z: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    """
    Delta: [P, N]
    Z: [r, N]
    return U: [P, r]
    """
    r = Z.shape[0]
    ZZt = Z @ Z.T
    reg = ridge_lambda * torch.eye(r, device=Z.device, dtype=Z.dtype)
    U = Delta @ Z.T @ torch.linalg.inv(ZZt + reg)
    return U


@torch.no_grad()
def reconstruction_errors(Delta: torch.Tensor, U: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    recon = U @ Z
    return torch.sum((Delta - recon) ** 2, dim=0)


@torch.no_grad()
def solve_subspace_coordinate(U: torch.Tensor, delta_vec: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    r = U.shape[1]
    reg = ridge_lambda * torch.eye(r, device=U.device, dtype=U.dtype)
    z = torch.linalg.solve(U.T @ U + reg, U.T @ delta_vec)
    return z


# =========================================================
# Aggregation
# =========================================================
@torch.no_grad()
def pairwise_sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A2 = (A ** 2).sum(dim=1, keepdim=True)
    B2 = (B ** 2).sum(dim=1, keepdim=True).T
    return A2 + B2 - 2.0 * (A @ B.T)


@torch.no_grad()
def kmeans_torch(
    X: torch.Tensor,
    n_clusters: int,
    n_iters: int = 20,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, _ = X.shape
    K = min(max(1, n_clusters), N)

    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    perm = torch.randperm(N, generator=g, device=X.device)
    centers = X[perm[:K]].clone()

    assign = torch.zeros(N, dtype=torch.long, device=X.device)

    for _ in range(n_iters):
        dists = pairwise_sq_dists(X, centers)
        new_assign = dists.argmin(dim=1)

        new_centers = []
        for k in range(K):
            mask = (new_assign == k)
            if mask.any():
                new_centers.append(X[mask].mean(dim=0))
            else:
                ridx = torch.randint(0, N, (1,), generator=g, device=X.device).item()
                new_centers.append(X[ridx].clone())
        new_centers = torch.stack(new_centers, dim=0)

        if torch.equal(new_assign, assign):
            centers = new_centers
            break

        assign = new_assign
        centers = new_centers

    return assign, centers


@torch.no_grad()
def softmax_weights_from_distance(
    X: torch.Tensor,
    center: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    d2 = ((X - center[None, :]) ** 2).sum(dim=1)
    w = torch.softmax(-d2 / max(tau, 1e-6), dim=0)
    return w


@torch.no_grad()
def clustered_task_aware_aggregate(
    Delta: torch.Tensor,
    Z_raw: torch.Tensor,
    assign: torch.Tensor,
    client_sizes: torch.Tensor,
    tau_intra: float = 1.0,
    tau_inter: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = Delta.device
    unique_clusters = torch.unique(assign).tolist()
    global_task_center = Z_raw.mean(dim=0)

    cluster_deltas = []
    cluster_centers = []
    cluster_sizes = []
    cluster_member_counts = []

    for k in unique_clusters:
        idx = torch.where(assign == k)[0]
        if idx.numel() == 0:
            continue

        Zk = Z_raw[idx]
        Dk = Delta[:, idx]
        nk_sizes = client_sizes[idx]

        center_k = Zk.mean(dim=0)

        w_dist = softmax_weights_from_distance(Zk, center_k, tau=tau_intra)
        w_size = nk_sizes / (nk_sizes.sum() + 1e-12)
        w = w_dist * w_size
        w = w / (w.sum() + 1e-12)

        delta_k = Dk @ w

        cluster_deltas.append(delta_k)
        cluster_centers.append(center_k)
        cluster_sizes.append(float(nk_sizes.sum().item()))
        cluster_member_counts.append(int(idx.numel()))

    cluster_deltas = torch.stack(cluster_deltas, dim=1)
    cluster_centers = torch.stack(cluster_centers, dim=0)
    cluster_sizes = torch.tensor(cluster_sizes, dtype=torch.float32, device=device)

    dist2_to_global = ((cluster_centers - global_task_center[None, :]) ** 2).sum(dim=1)
    w_inter_dist = torch.softmax(-dist2_to_global / max(tau_inter, 1e-6), dim=0)
    w_inter_size = cluster_sizes / (cluster_sizes.sum() + 1e-12)
    w_inter = w_inter_dist * w_inter_size
    w_inter = w_inter / (w_inter.sum() + 1e-12)

    delta_agg = cluster_deltas @ w_inter

    stats = {
        "cluster_centers": cluster_centers.detach().cpu(),
        "cluster_sizes": cluster_sizes.detach().cpu(),
        "cluster_member_counts": torch.tensor(cluster_member_counts, dtype=torch.long),
        "w_inter": w_inter.detach().cpu(),
    }
    return delta_agg, stats


@torch.no_grad()
def weighted_fedavg_delta(
    Delta: torch.Tensor,          # [P, N]
    client_sizes: torch.Tensor,   # [N]
) -> torch.Tensor:
    w = client_sizes.to(Delta.device).float()
    w = w / (w.sum() + 1e-12)
    return Delta @ w


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    ap.add_argument("--domain", type=str, required=True, choices=["gsm8k", "dolly", "mbpp", "mmlu", "bbh"])
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--anchor_file", type=str, required=True)
    ap.add_argument("--n_anchors", type=int, default=80)
    ap.add_argument("--anchor_groups", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_tag", type=str, default="dirichlet_hash_alpha0.1")
    ap.add_argument("--clients", type=int, default=10)

    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--local_steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--min_ans_tokens", type=int, default=8)
    ap.add_argument("--local_batch_size", type=int, default=16)
    ap.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--grad_clip", type=float, default=2.0)
    ap.add_argument("--train_domain", type=str, default="auto", help="Training split domain. Use auto to train GSM8K/MMLU/BBH from alpaca_gpt4 splits and Dolly from dolly splits.")
    ap.add_argument("--dp_epsilon", type=float, default=6.0, help="Experimental DP epsilon label. Set <=0 to disable DP perturbation.")
    ap.add_argument("--dp_clip_norm", type=float, default=1.0)
    ap.add_argument("--dp_noise_multiplier", type=float, default=1.0, help="Gaussian noise multiplier for uploaded LoRA-B deltas. Noise std is noise_multiplier * dp_clip_norm / dp_epsilon.")

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--probe_layers", type=str, default="all")
    ap.add_argument("--router_temp", type=float, default=1.0)
    ap.add_argument("--no_feature_norm", action="store_true")
    ap.add_argument("--task_dim", type=int, default=8)
    ap.add_argument("--ridge_lambda", type=float, default=1e-4)

    ap.add_argument("--num_clusters", type=int, default=3)
    ap.add_argument("--kmeans_iters", type=int, default=20)
    ap.add_argument("--tau_intra", type=float, default=1.0)
    ap.add_argument("--tau_inter", type=float, default=1.0)

    ap.add_argument("--beta_router", type=float, default=0.5)

    ap.add_argument("--show_probe_progress", action="store_true")
    ap.add_argument("--disable_tqdm", action="store_true")

    ap.add_argument("--eval_n", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--eval_split", type=str, default="test", choices=["val", "test"])

    ap.add_argument("--metrics_file", type=str, default="logs/fedtar_metrics.jsonl")

    ap.add_argument("--wandb_project", type=str, default="fl-lora-moe-noniid0.1")
    ap.add_argument("--wandb_run_name", type=str, default="fedtar_bonly_fixedA")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    if args.eval_every <= 0:
        raise ValueError("--eval_every must be >= 1")
    if args.num_clusters <= 0:
        raise ValueError("--num_clusters must be >= 1")
    if not (0.0 <= args.beta_router <= 1.0):
        raise ValueError("--beta_router must be in [0, 1]")

    if args.domain in {"mmlu", "bbh"} and args.min_ans_tokens > 1:
        print(f"[WARN] {args.domain.upper()} detected: overriding min_ans_tokens from {args.min_ans_tokens} to 1")
        args.min_ans_tokens = 1

    set_seed(args.seed)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    run_name = args.wandb_run_name.strip() or f"fedtar_bonly_fixedA_{args.domain}_seed{args.seed}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    anchors_all = load_jsonl(args.anchor_file)
    anchors = anchors_all[:args.n_anchors]
    anchor_groups = split_anchor_groups(anchors, args.anchor_groups)

    if args.train_domain == "auto":
        # Experimental setup: GSM8K/MMLU/BBH follow the general instruction-tuning protocol
        # and use Alpaca-GPT-4-style federated training splits; Dolly uses its own
        # task-category-based federated split.
        train_domain = "dolly" if args.domain == "dolly" else "alpaca_gpt4"
    else:
        train_domain = args.train_domain

    split_root = Path(args.data_dir) / "fl_splits" / f"seed{args.seed}" / args.split_tag / train_domain
    fixed_eval_dir = Path(args.data_dir) / "fixed_eval" / f"seed{args.seed}"

    client_rows = {cid: load_jsonl(split_root / f"client_{cid:02d}" / "train.jsonl") for cid in range(args.clients)}
    client_sizes = torch.tensor(
        [max(len(client_rows[cid]), 1) for cid in range(args.clients)],
        dtype=torch.float32
    )

    tokenizer, global_model, target_modules = build_lora_model(
        model_name=args.model_name,
        dtype=args.dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    global_A_sd = clone_sd(get_lora_A_state_dict(global_model))
    global_B_sd = clone_sd(get_lora_state_dict(global_model, only_lora_B=True))
    global_lora_sd = clone_sd(global_B_sd)

    set_partial_state_dict(global_model, global_A_sd)
    set_lora_state_dict(global_model, global_lora_sd)
    freeze_lora_A_unfreeze_lora_B(global_model)

    num_experts = getattr(global_model.config, "num_experts", None)
    if num_experts is None:
        raise RuntimeError("Cannot find config.num_experts for this MoE model.")

    layers_to_probe = get_probe_layers_from_model(global_model, args.probe_layers)

    print(f"[INIT] method=FedTAR")
    print(f"[INIT] eval_domain={args.domain}")
    print(f"[INIT] train_domain={train_domain}")
    print(f"[INIT] clients={args.clients}, rounds={args.rounds}")
    print(f"[INIT] task_dim={args.task_dim}, ridge_lambda={args.ridge_lambda}")
    print(f"[INIT] num_clusters={args.num_clusters}, tau_intra={args.tau_intra}, tau_inter={args.tau_inter}")
    print(f"[INIT] beta_router={args.beta_router}")
    print(f"[INIT] probe_layers={layers_to_probe}")
    print(f"[INIT] num_experts={num_experts}")
    print(f"[INIT] anchors={len(anchors)} groups={len(anchor_groups)}")
    print(f"[INIT] LoRA target modules={target_modules}")
    print(f"[INIT] FedTAR B-only with fixed A = True")
    print(f"[INIT] local_batch_size={args.local_batch_size}, optimizer={args.optimizer}, grad_clip={args.grad_clip}")
    print(f"[INIT] dp_epsilon={args.dp_epsilon}, dp_clip_norm={args.dp_clip_norm}, dp_noise_multiplier={args.dp_noise_multiplier}")
    print(f"[INIT] min_ans_tokens={args.min_ans_tokens}")
    print(f"[INIT] eval_every={args.eval_every}, eval_n={args.eval_n}, eval_split={args.eval_split}")
    print(f"[INIT] metrics_file={args.metrics_file}")

    mname, mval = eval_single_domain(
        args.domain,
        tokenizer,
        global_model,
        fixed_eval_dir,
        n_samples=args.eval_n,
        split=args.eval_split,
    )
    print(f"[INIT-EVAL] {args.domain} {mname}@{args.eval_n}={mval:.4f}")
    wandb.log({f"global/{args.domain}_{mname}": float(mval), "round": 0})

    append_jsonl(args.metrics_file, {
        "round": 0,
        "domain": args.domain,
        "split": args.eval_split,
        "metric_name": mname,
        "metric_value": float(mval),
        "mean_client_loss": None,
        "delta_fedavg_norm": None,
        "delta_router_norm": None,
        "delta_global_norm": None,
        "beta_router": float(args.beta_router),
        "method": "FedTAR",
        "train_domain": train_domain,
        "local_batch_size": int(args.local_batch_size),
        "optimizer": args.optimizer,
        "grad_clip": float(args.grad_clip),
        "dp_epsilon": float(args.dp_epsilon),
        "dp_clip_norm": float(args.dp_clip_norm),
        "dp_noise_multiplier": float(args.dp_noise_multiplier),
    })

    t_all = time.time()
    round_durations: List[float] = []

    round_iter = range(1, args.rounds + 1)
    if not args.disable_tqdm:
        round_iter = tqdm(round_iter, desc="FL Rounds")

    for rnd in round_iter:
        round_start = time.time()
        print(f"\n===== Round {rnd}/{args.rounds} =====")

        client_deltas: List[Dict[str, torch.Tensor]] = []
        client_losses: List[float] = []
        routing_sigs: List[torch.Tensor] = []
        valid_opt_steps_all: List[int] = []
        skipped_micro_all: List[int] = []

        client_iter = range(args.clients)
        if not args.disable_tqdm:
            client_iter = tqdm(client_iter, desc=f"Round {rnd} Clients", leave=False)

        for cid in client_iter:
            client_start = time.time()
            print(f"  client_{cid:02d}")

            _, local_model, _ = build_lora_model(
                model_name=args.model_name,
                dtype=args.dtype,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )

            set_partial_state_dict(local_model, global_A_sd)
            set_lora_state_dict(local_model, global_lora_sd)
            freeze_lora_A_unfreeze_lora_B(local_model)

            loss, valid_opt_steps, skipped_micro = local_train_rows_safe(
                tokenizer=tokenizer,
                model=local_model,
                rows=client_rows[cid],
                steps=args.local_steps,
                lr=args.lr,
                max_len=args.max_len,
                min_ans_tokens=args.min_ans_tokens,
                local_batch_size=args.local_batch_size,
                optimizer_name=args.optimizer,
                grad_clip=args.grad_clip,
                log_every=50,
            )
            client_losses.append(float(loss))
            valid_opt_steps_all.append(int(valid_opt_steps))
            skipped_micro_all.append(int(skipped_micro))

            local_sd = clone_sd(get_lora_state_dict(local_model, only_lora_B=True))
            delta_i = diff_sd(local_sd, global_lora_sd)
            delta_i = apply_dp_to_state_dict(
                delta_i,
                clip_norm=args.dp_clip_norm,
                epsilon=args.dp_epsilon,
                noise_multiplier=args.dp_noise_multiplier,
            )
            client_deltas.append(delta_i)

            x_i = compute_routing_signature(
                model=local_model,
                tokenizer=tokenizer,
                anchor_groups=anchor_groups,
                layers_to_probe=layers_to_probe,
                max_len=min(args.max_len, 256),
                num_experts=num_experts,
                device=local_model.device,
                temp=args.router_temp,
                normalize_feature=(not args.no_feature_norm),
                show_probe_progress=args.show_probe_progress and (not args.disable_tqdm),
            )
            routing_sigs.append(x_i.cpu())

            client_time = time.time() - client_start
            if not args.disable_tqdm:
                client_iter.set_postfix({
                    "loss": f"{loss:.3f}",
                    "valid": valid_opt_steps,
                    "skip": skipped_micro,
                    "client_t": format_seconds(client_time),
                })

            del local_model
            torch.cuda.empty_cache()

        X = torch.stack(routing_sigs, dim=0).float()
        Z_client, _, _ = pca_project(X, proj_dim=args.task_dim)
        Z = Z_client.T.contiguous()

        template, keys = state_dict_template_and_keys(client_deltas)
        Delta_vecs = [flatten_state_dict(sd, template, keys) for sd in client_deltas]
        Delta = torch.stack(Delta_vecs, dim=1).float()

        U = fit_task_subspace_ridge(Delta=Delta, Z=Z, ridge_lambda=args.ridge_lambda)
        recon_err = reconstruction_errors(Delta, U, Z)

        assign, centers = kmeans_torch(
            Z_client,
            n_clusters=args.num_clusters,
            n_iters=args.kmeans_iters,
            seed=args.seed + rnd,
        )

        delta_agg_vec, agg_stats = clustered_task_aware_aggregate(
            Delta=Delta,
            Z_raw=Z_client,
            assign=assign,
            client_sizes=client_sizes.to(Delta.device),
            tau_intra=args.tau_intra,
            tau_inter=args.tau_inter,
        )

        z_agg = solve_subspace_coordinate(U, delta_agg_vec, args.ridge_lambda)
        delta_router_vec = U @ z_agg

        delta_fedavg_vec = weighted_fedavg_delta(
            Delta=Delta,
            client_sizes=client_sizes.to(Delta.device),
        )

        beta = float(args.beta_router)
        delta_global_vec = (1.0 - beta) * delta_fedavg_vec + beta * delta_router_vec
        delta_global = unflatten_vector_to_state_dict(delta_global_vec, template, keys)

        global_lora_sd = add_sd(global_lora_sd, delta_global)
        set_partial_state_dict(global_model, global_A_sd)
        set_lora_state_dict(global_model, global_lora_sd)
        freeze_lora_A_unfreeze_lora_B(global_model)

        mean_loss = sum(client_losses) / max(len(client_losses), 1)
        round_time = time.time() - round_start
        round_durations.append(round_time)
        avg_round_time = sum(round_durations) / len(round_durations)
        eta_sec = (args.rounds - rnd) * avg_round_time

        mean_recon = float(recon_err.mean().item())
        max_recon = float(recon_err.max().item())
        delta_agg_norm = safe_norm(delta_agg_vec)
        z_agg_norm = safe_norm(z_agg)
        delta_router_norm = safe_norm(delta_router_vec)
        delta_fedavg_norm = safe_norm(delta_fedavg_vec)
        delta_global_norm = safe_norm(delta_global_vec)
        mean_client_delta_norm = float(torch.tensor([safe_norm(v) for v in Delta_vecs]).mean().item())

        unique_clusters, counts = torch.unique(assign.cpu(), return_counts=True)
        cluster_count_dict = {int(k.item()): int(v.item()) for k, v in zip(unique_clusters, counts)}

        do_eval = (rnd % args.eval_every == 0) or (rnd == args.rounds)

        if do_eval:
            mname, mval = eval_single_domain(
                args.domain,
                tokenizer,
                global_model,
                fixed_eval_dir,
                n_samples=args.eval_n,
                split=args.eval_split,
            )
            print(f"[ROUND {rnd}] mean_client_loss={mean_loss:.4f} | {args.domain} {mname}@{args.eval_n}={mval:.4f}")
        else:
            mname, mval = None, None
            print(f"[ROUND {rnd}] mean_client_loss={mean_loss:.4f} | eval=skip")

        print(f"[ROUND {rnd}] mean_recon={mean_recon:.4f} | max_recon={max_recon:.4f}")
        print(f"[ROUND {rnd}] mean_client_delta_norm={mean_client_delta_norm:.6f}")
        print(f"[ROUND {rnd}] delta_fedavg_norm={delta_fedavg_norm:.6f} | delta_router_norm={delta_router_norm:.6f} | delta_global_norm={delta_global_norm:.6f}")
        print(f"[ROUND {rnd}] delta_agg_norm={delta_agg_norm:.6f} | z_agg_norm={z_agg_norm:.6f}")
        print(f"[ROUND {rnd}] clusters={cluster_count_dict}")
        print(f"[ROUND {rnd}] inter_weights={agg_stats['w_inter'].tolist()}")
        print(f"[ROUND {rnd}] valid_opt_steps={valid_opt_steps_all} | skipped_micro={skipped_micro_all}")
        print(f"[ROUND {rnd}] time={format_seconds(round_time)} | ETA={format_seconds(eta_sec)}")

        if not args.disable_tqdm:
            postfix = {
                "loss": f"{mean_loss:.3f}",
                "dglobal": f"{delta_global_norm:.2e}",
                "eta": format_seconds(eta_sec),
            }
            if do_eval and mval is not None:
                postfix["metric"] = f"{mval:.4f}"
            round_iter.set_postfix(postfix)

        log_dict = {
            "round": rnd,
            "time/round_sec": float(round_time),
            "time/round_min": float(round_time / 60.0),
            "time/elapsed_min": float((time.time() - t_all) / 60.0),
            "time/eta_sec": float(eta_sec),

            "train/mean_client_loss": float(mean_loss),
            "train/mean_valid_opt_steps": float(sum(valid_opt_steps_all) / max(len(valid_opt_steps_all), 1)),
            "train/mean_skipped_micro": float(sum(skipped_micro_all) / max(len(skipped_micro_all), 1)),
            "train/local_batch_size": float(args.local_batch_size),
            "train/grad_clip": float(args.grad_clip),
            "privacy/dp_epsilon": float(args.dp_epsilon),
            "privacy/dp_clip_norm": float(args.dp_clip_norm),
            "privacy/dp_noise_multiplier": float(args.dp_noise_multiplier),

            "subspace/mean_recon_error": mean_recon,
            "subspace/max_recon_error": max_recon,
            "subspace/z_agg_norm": z_agg_norm,

            "update/mean_client_delta_norm": mean_client_delta_norm,
            "update/delta_fedavg_norm": delta_fedavg_norm,
            "update/delta_agg_norm": delta_agg_norm,
            "update/delta_router_norm": delta_router_norm,
            "update/delta_global_norm": delta_global_norm,
            "update/beta_router": float(args.beta_router),

            "cluster/num_effective_clusters": float(len(cluster_count_dict)),
        }

        for k, v in cluster_count_dict.items():
            log_dict[f"cluster/size_{k}"] = float(v)

        w_inter = agg_stats["w_inter"].tolist()
        for i, w in enumerate(w_inter):
            log_dict[f"cluster/inter_weight_{i}"] = float(w)

        if do_eval and mname is not None:
            log_dict[f"global/{args.domain}_{mname}"] = float(mval)

        wandb.log(log_dict)

        metrics_row = {
            "round": rnd,
            "domain": args.domain,
            "split": args.eval_split,
            "metric_name": mname,
            "metric_value": None if mval is None else float(mval),

            "mean_client_loss": float(mean_loss),
            "mean_valid_opt_steps": float(sum(valid_opt_steps_all) / max(len(valid_opt_steps_all), 1)),
            "mean_skipped_micro": float(sum(skipped_micro_all) / max(len(skipped_micro_all), 1)),
            "local_batch_size": int(args.local_batch_size),
            "optimizer": args.optimizer,
            "grad_clip": float(args.grad_clip),
            "train_domain": train_domain,
            "dp_epsilon": float(args.dp_epsilon),
            "dp_clip_norm": float(args.dp_clip_norm),
            "dp_noise_multiplier": float(args.dp_noise_multiplier),

            "mean_recon_error": mean_recon,
            "max_recon_error": max_recon,
            "z_agg_norm": z_agg_norm,

            "mean_client_delta_norm": mean_client_delta_norm,
            "delta_fedavg_norm": delta_fedavg_norm,
            "delta_agg_norm": delta_agg_norm,
            "delta_router_norm": delta_router_norm,
            "delta_global_norm": delta_global_norm,
            "beta_router": float(args.beta_router),

            "round_sec": float(round_time),
            "elapsed_min": float((time.time() - t_all) / 60.0),
            "eta_sec": float(eta_sec),

            "cluster_counts": cluster_count_dict,
            "inter_weights": agg_stats["w_inter"].tolist(),
            "valid_opt_steps_all": valid_opt_steps_all,
            "skipped_micro_all": skipped_micro_all,
            "client_losses": client_losses,
        }
        append_jsonl(args.metrics_file, metrics_row)

    wandb.finish()


if __name__ == "__main__":
    main()