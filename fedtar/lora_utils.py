from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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

