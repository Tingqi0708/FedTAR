from typing import Dict, Tuple

import torch

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

