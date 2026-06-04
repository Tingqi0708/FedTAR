from typing import Dict, List, Tuple

import torch

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

