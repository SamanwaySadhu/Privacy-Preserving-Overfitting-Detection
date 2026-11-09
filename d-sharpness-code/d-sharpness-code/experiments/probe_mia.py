#!/usr/bin/env python3
"""Membership inference attack on a single VGG-16-BN trained on CIFAR-10.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from df_pair_common import autocast_ctx, resolve_device  # noqa: E402

DFLAT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(DFLAT_ROOT / "df_utils"))
import df as df_module  # noqa: E402
import sam_train  # noqa: E402


def _load_or_train_model(args: argparse.Namespace, device: torch.device, train_ds) -> nn.Module:
    model = sam_train.prepare_model(args.model_name, args.dataset, int(args.seed)).to(device)
    if args.ckpt is not None and Path(args.ckpt).exists():
        payload = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"[mia] loaded checkpoint from {args.ckpt}", flush=True)
        return model

    print(f"[mia] no --ckpt provided, training fresh for up to {args.epochs} epochs", flush=True)
    loader = DataLoader(
        train_ds, batch_size=int(args.train_batch_size), shuffle=True,
        num_workers=int(args.train_num_workers), pin_memory=torch.cuda.is_available(),
    )
    opt = torch.optim.SGD(model.parameters(), lr=float(args.lr),
                          momentum=float(args.momentum), weight_decay=float(args.weight_decay))
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        # Enter autocast once per epoch; per-batch entry cost would dominate B=128 ops on MPS.
        with autocast_ctx(device, "fp32"):
            for batch in loader:
                xb, yb = batch[0].to(device, non_blocking=torch.cuda.is_available()), batch[1].to(device, non_blocking=torch.cuda.is_available()).long()
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        idx = np.random.default_rng(ep).choice(len(train_ds), size=min(1024, len(train_ds)), replace=False)
        with torch.no_grad(), autocast_ctx(device, "fp32"):
            xb = torch.stack([train_ds[i][0] for i in idx]).to(device)
            yb = torch.tensor([train_ds[i][1] for i in idx], device=device, dtype=torch.long)
            loss_est = float(F.cross_entropy(model(xb), yb).item())
        print(f"[mia] epoch {ep:03d} sample_train_loss={loss_est:.4f}", flush=True)
        if loss_est <= float(args.stop_train_loss):
            print(f"[mia] reached stop-train-loss={args.stop_train_loss}; stopping", flush=True)
            break

    if args.save_ckpt is not None:
        Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, str(args.save_ckpt))
        print(f"[mia] saved checkpoint to {args.save_ckpt}", flush=True)
    return model


@torch.no_grad()
def _per_sample_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, device: torch.device) -> float:
    # Autocast is entered ONCE around the calling loop in `_score_set`, not here.
    model.eval()
    logits = model(x.to(device).unsqueeze(0))
    return float(F.cross_entropy(logits, y.to(device).long().unsqueeze(0)).item())


def _per_sample_static_sharpness(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, device: torch.device,
    flat_r: float, flat_delta: float, adaptive: bool, flat_norm: str,
) -> float:
    batch = (x.unsqueeze(0), torch.tensor([int(y.item())], dtype=torch.long))
    # Autocast is entered ONCE around the calling loop in `_score_set`, not here.
    stats = df_module.flatness_on_batch(
        model, batch, device, norm=str(flat_norm), adaptive=bool(adaptive),
        flat_r=float(flat_r), flat_delta=float(flat_delta),
        freeze_bn=True, enable_grads=True,
    )
    return float(stats["flat_primal"])


def _per_sample_directional_scores(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, device: torch.device,
    probe_steps: int, sam_rho: float, sam_lr: float, sam_weight_decay: float,
    sam_momentum: float, sam_adaptive: bool, sam_base_opt: str,
    flat_r: float, flat_delta: float, flat_norm: str,
) -> Dict[str, float]:
    """T-step DF probe with B=1; returns every series score in one pass.

    The probe trajectory is computed once; deriving all 9 DEFAULT_SCORE_COLUMNS
    from it is essentially free (no extra forward/backward).
    """
    probe_model = copy.deepcopy(model).to(device)
    params = [p for p in probe_model.parameters() if p.requires_grad]
    base_cls = torch.optim.AdamW if str(sam_base_opt).lower() == "adamw" else torch.optim.SGD
    base_kwargs = {"lr": float(sam_lr), "weight_decay": float(sam_weight_decay)}
    if base_cls is torch.optim.SGD:
        base_kwargs["momentum"] = float(sam_momentum)
    sam_opt = sam_train.SAM(
        params, base_cls, rho=float(sam_rho), adaptive=bool(sam_adaptive), **base_kwargs,
    )
    batch = (x.unsqueeze(0), torch.tensor([int(y.item())], dtype=torch.long))

    flat_primals = []
    losses = []
    # Autocast is entered ONCE around the calling loop in `_score_set`, not here.
    for _ in range(int(probe_steps)):
        stats = df_module.flatness_on_batch(
            probe_model, batch, device, norm=str(flat_norm), adaptive=bool(sam_adaptive),
            flat_r=float(flat_r), flat_delta=float(flat_delta),
            freeze_bn=True, params=params, enable_grads=True,
        )
        flat_primals.append(float(stats["flat_primal"]))
        losses.append(float(stats["mb_loss"]))
        df_module.sam_update_on_batch(probe_model, sam_opt, batch, device)

    return df_module.compute_scores_for_series(np.asarray(flat_primals), np.asarray(losses))


def _score_set(
    model, dataset, indices, *, device, args, df_scores: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Per-sample (loss, static_sharpness, {variant: directional}) over `indices`.
    """
    losses: List[float] = []
    statics: List[float] = []
    directionals: Dict[str, List[float]] = {v: [] for v in df_scores}
    total = len(indices)
    with autocast_ctx(device, "fp32"):
        for i, idx in enumerate(indices):
            x, y_int = dataset[int(idx)]
            y = torch.tensor(int(y_int), dtype=torch.long)
            losses.append(_per_sample_loss(model, x, y, device=device))
            statics.append(_per_sample_static_sharpness(
                model, x, y, device=device,
                flat_r=float(args.flat_r), flat_delta=float(args.flat_delta),
                adaptive=bool(args.sam_adaptive), flat_norm=str(args.flat_norm),
            ))
            per_variant = _per_sample_directional_scores(
                model, x, y, device=device,
                probe_steps=int(args.probe_steps),
                sam_rho=float(args.sam_rho), sam_lr=float(args.sam_lr),
                sam_weight_decay=float(args.sam_weight_decay), sam_momentum=float(args.sam_momentum),
                sam_adaptive=bool(args.sam_adaptive), sam_base_opt=str(args.sam_base_opt),
                flat_r=float(args.flat_r), flat_delta=float(args.flat_delta),
                flat_norm=str(args.flat_norm),
            )
            for v in df_scores:
                directionals[v].append(float(per_variant.get(v, float("nan"))))
            if (i + 1) % max(1, total // 20) == 0:
                print(f"[mia] scored {i + 1}/{total}", flush=True)
    return (
        np.asarray(losses),
        np.asarray(statics),
        {v: np.asarray(vals) for v, vals in directionals.items()},
    )


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC where labels in {0,1} and higher score => label==1."""
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    # Rank-based AUC with tie correction.
    ranks = np.empty_like(s, dtype=float)
    n = len(s)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _best_threshold_acc(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Return (best_acc, advantage TPR-FPR at the best-advantage threshold)."""
    order = np.argsort(scores, kind="mergesort")[::-1]
    s = scores[order]
    y = labels[order]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan"), float("nan")
    tp = np.cumsum(y == 1).astype(np.float64)
    fp = np.cumsum(y == 0).astype(np.float64)
    tpr = tp / pos
    fpr = fp / neg
    adv = tpr - fpr
    best = int(np.argmax(adv))
    # Threshold value: anything with score >= s[best] is predicted positive.
    thr = float(s[best])
    preds = (scores >= thr).astype(int)
    acc = float((preds == labels).mean())
    return acc, float(adv[best])


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=DFLAT_ROOT / "data")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--model-name", type=str, default="vgg16_bn")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--ckpt", type=Path, default=DFLAT_ROOT / "artifacts" / "mia_trained_model.pt",
                   help="If provided, load weights from this checkpoint and skip training.")
    p.add_argument("--save-ckpt", type=Path, default=DFLAT_ROOT / "artifacts" / "mia_trained_model.pt",
                   help="If provided and we trained from scratch, save the final weights here.")

    # Training (only used when --ckpt is not provided).
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--train-num-workers", type=int, default=2)
    p.add_argument("--stop-train-loss", type=float, default=0.01)

    # Attack setup.
    p.add_argument("--n-attack", type=int, default=200, help="Members and non-members each.")

    # Probe knobs.
    p.add_argument("--probe-steps", type=int, default=32, help="T in directional sharpness.")
    p.add_argument("--sam-rho", type=float, default=0.05)
    p.add_argument("--sam-lr", type=float, default=1e-4)
    p.add_argument("--sam-weight-decay", type=float, default=0.0)
    p.add_argument("--sam-momentum", type=float, default=0.9)
    p.add_argument("--sam-adaptive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sam-base-opt", type=str, default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--flat-r", type=float, default=1.0)
    p.add_argument("--flat-delta", type=float, default=1.0)
    p.add_argument("--flat-norm", type=str, default="l1", choices=["l1", "l2"])
    p.add_argument("--df-scores", type=str, default=",".join(df_module.SCORE_COLUMNS),
                   help=("Comma-separated DF series scores to report. Defaults to all 9 variants "
                         "(FLAT_STD/VAR/RANGE, FLAT_OVER_LOSS_*, FLAT2_OVER_LOSS_*)."))
    p.add_argument("--device-index", type=int, default=0)
    args = p.parse_args(argv)

    df_scores: List[str] = [s.strip() for s in str(args.df_scores).split(",") if s.strip()]
    unknown = [s for s in df_scores if s not in df_module.SCORE_COLUMNS]
    if unknown:
        raise SystemExit(f"--df-scores has unknown variant(s): {unknown}. "
                         f"Valid: {list(df_module.SCORE_COLUMNS)}")

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = DFLAT_ROOT / "artifacts" / f"mia_{stamp}"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    sam_train.set_seed(int(args.seed))
    device = resolve_device(prefer_index=int(args.device_index), allow_cpu_fallback=True)
    print(f"[mia] device={device}", flush=True)

    # Probing/loss is done on un-augmented inputs to keep per-sample scores deterministic.
    train_aug_ds, test_ds = sam_train.prepare_dataset(
        args.dataset, root=str(args.data_root), normalize=True, randomize=True, cutout=False,
    )
    train_eval_ds, _ = sam_train.prepare_dataset(
        args.dataset, root=str(args.data_root), normalize=True, randomize=False, cutout=False,
    )

    model = _load_or_train_model(args, device, train_aug_ds)
    model.to(device).eval()

    rng = np.random.default_rng(int(args.seed))
    n_members = min(int(args.n_attack), len(train_eval_ds))
    n_nonmembers = min(int(args.n_attack), len(test_ds))
    member_idx = rng.choice(len(train_eval_ds), n_members, replace=False)
    nonmember_idx = rng.choice(len(test_ds), n_nonmembers, replace=False)
    print(f"[mia] scoring {n_members} members + {n_nonmembers} non-members", flush=True)

    loss_m, stat_m, dir_m = _score_set(
        model, train_eval_ds, member_idx, device=device, args=args, df_scores=df_scores,
    )
    loss_n, stat_n, dir_n = _score_set(
        model, test_ds, nonmember_idx, device=device, args=args, df_scores=df_scores,
    )

    labels = np.concatenate([np.ones_like(loss_m), np.zeros_like(loss_n)]).astype(int)
    # Higher = more member-like. Loss is lower for members, so negate it.
    scores: Dict[str, np.ndarray] = {
        "loss":             np.concatenate([-loss_m, -loss_n]),
        "static_sharpness": np.concatenate([-stat_m, -stat_n]),
    }
    for v in df_scores:
        scores[f"directional[{v}]"] = np.concatenate([-dir_m[v], -dir_n[v]])

    rows = []
    print(f"\n{'Metric':<28}{'ROC-AUC':>10}{'Acc':>10}{'Advantage':>12}{'  Direction'}")
    print("-" * 70)
    for name, sc in scores.items():
        auc = _roc_auc(sc, labels)
        acc, adv = _best_threshold_acc(sc, labels)
        auc_flip = _roc_auc(-sc, labels)
        acc_flip, adv_flip = _best_threshold_acc(-sc, labels)
        if auc_flip > auc:
            auc, acc, adv = auc_flip, acc_flip, adv_flip
            sign = "low-is-member"
        else:
            sign = "high-is-member"
        print(f"{name:<28}{auc:>10.4f}{acc:>10.4f}{adv:>12.4f}  {sign}")
        rows.append({"metric": name, "auc": auc, "acc": acc, "advantage": adv, "direction": sign})

    import pandas as pd  # local import keeps cold-start light
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    payload: Dict[str, np.ndarray] = {
        "member_loss": loss_m, "member_static": stat_m,
        "nonmember_loss": loss_n, "nonmember_static": stat_n,
        "member_idx": member_idx, "nonmember_idx": nonmember_idx,
    }
    for v in df_scores:
        payload[f"member_directional_{v}"] = dir_m[v]
        payload[f"nonmember_directional_{v}"] = dir_n[v]
    np.savez(out_dir / "scores.npz", **payload)
    print(f"\n[mia] artifacts -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
