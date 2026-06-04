import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List

import torch
import wandb
from tqdm import tqdm

from fedtar.aggregation import clustered_task_aware_aggregate, kmeans_torch, weighted_fedavg_delta
from fedtar.evaluation import eval_single_domain
from fedtar.local_training import apply_dp_to_state_dict, local_train_rows_safe
from fedtar.lora_utils import (
    add_sd,
    build_lora_model,
    clone_sd,
    diff_sd,
    freeze_lora_A_unfreeze_lora_B,
    get_lora_A_state_dict,
    get_lora_state_dict,
    set_lora_state_dict,
    set_partial_state_dict,
)
from fedtar.routing import compute_routing_signature, get_probe_layers_from_model, split_anchor_groups
from fedtar.subspace import (
    fit_task_subspace_ridge,
    flatten_state_dict,
    pca_project,
    reconstruction_errors,
    solve_subspace_coordinate,
    state_dict_template_and_keys,
    unflatten_vector_to_state_dict,
)
from fedtar.utils import append_jsonl, format_seconds, load_jsonl, safe_norm, set_seed

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

