from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

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

