#!/usr/bin/env python3
"""Benign-vs-noisy-labels pair: clean CIFAR-10 vs `p` random label corruption.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from df_pair_common import (  # noqa: E402
    add_probe_sweep_args,
    all_sweeps_table,
    best_per_score,
    build_sweeps,
    evaluate_test_acc,
    load_train_result,
    make_cache_key,
    parse_df_scores,
    probe_and_score,
    resolve_device,
    save_train_result,
    static_comparison_table,
    static_sharpness_once,
    summarize_pair,
    train_sgd,
)

DFLAT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(DFLAT_ROOT / "df_utils"))
import sam_train  # noqa: E402


class NoisyLabelDataset(Dataset):
    def __init__(self, base: Dataset, noisy_targets: np.ndarray):
        self.base = base
        self.noisy_targets = np.asarray(noisy_targets, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.base))

    def __getitem__(self, index: int):
        item = self.base[index]
        return item[0], int(self.noisy_targets[index])


def build_noisy_targets(
    clean_targets: np.ndarray,
    *,
    noise_prob: float,
    num_classes: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    clean_targets = np.asarray(clean_targets, dtype=np.int64).copy()
    rng = np.random.default_rng(int(seed))
    if float(noise_prob) <= 0.0:
        return clean_targets, 0.0
    mask = rng.random(clean_targets.shape[0]) < float(noise_prob)
    idx = np.flatnonzero(mask)
    if idx.size > 0:
        sampled = rng.integers(0, max(1, int(num_classes) - 1), size=idx.size, dtype=np.int64)
        original = clean_targets[idx]
        sampled = sampled + (sampled >= original).astype(np.int64)
        clean_targets[idx] = sampled
    return clean_targets, float(mask.mean())


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=DFLAT_ROOT / "data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-name", type=str, default="vgg16_bn")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--noise-prob", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--train-num-workers", type=int, default=2)
    p.add_argument("--stop-train-acc", type=float, default=0.9)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=("cosine", "multistep", "none"),
                   help="SGD LR schedule: cosine annealing, multistep, or none (constant LR).")
    add_probe_sweep_args(p)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--ckpt-cache-dir", type=Path, default=DFLAT_ROOT / "artifacts" / "ckpts" / "noisy_pair",
                   help="Directory to cache trained-from-scratch checkpoints. Re-runs with matching args load from here.")
    p.add_argument("--ckpt-cache", action=argparse.BooleanOptionalAction, default=True,
                   help="If false, ignore the cache and always retrain.")
    args = p.parse_args(argv)

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = DFLAT_ROOT / "artifacts" / f"noisy_pair_{stamp}"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    sam_train.set_seed(int(args.seed))
    device = resolve_device(prefer_index=int(args.device_index), allow_cpu_fallback=True)
    print(f"[noisy_pair] device={device}", flush=True)

    train_aug_ds, test_ds = sam_train.prepare_dataset(
        args.dataset, root=str(args.data_root), normalize=True, randomize=True, cutout=False,
    )
    train_eval_ds, _ = sam_train.prepare_dataset(
        args.dataset, root=str(args.data_root), normalize=True, randomize=False, cutout=False,
    )

    clean_targets = np.asarray(getattr(train_eval_ds, "targets"), dtype=np.int64).copy()
    num_classes = int(np.max(clean_targets) + 1)
    noisy_targets, realized = build_noisy_targets(
        clean_targets,
        noise_prob=float(args.noise_prob),
        num_classes=num_classes,
        seed=int(args.seed) + 100_003,
    )
    noisy_train_aug_ds = NoisyLabelDataset(train_aug_ds, noisy_targets)
    noisy_train_eval_ds = NoisyLabelDataset(train_eval_ds, noisy_targets)
    print(f"[noisy_pair] noise realized={realized:.4f}", flush=True)

    cache_dir = Path(args.ckpt_cache_dir).expanduser().resolve() if args.ckpt_cache else None
    common_fields = {
        "model": args.model_name, "ds": args.dataset, "seed": args.seed,
        "lr": args.lr, "mom": args.momentum, "wd": args.weight_decay,
        "bs": args.train_batch_size, "ep": args.epochs, "stop": args.stop_train_acc,
        "sch": args.lr_schedule,
    }

    benign_key = make_cache_key("benign", common_fields)
    benign_ckpt = (cache_dir / f"{benign_key}.pt") if cache_dir is not None else None
    if benign_ckpt is not None and benign_ckpt.exists():
        print(f"[benign] loading cached ckpt from {benign_ckpt}", flush=True)
        benign_template = sam_train.prepare_model(args.model_name, args.dataset, int(args.seed))
        benign_res = load_train_result(benign_ckpt, benign_template, device=device)
    else:
        sam_train.set_seed(int(args.seed))
        benign_model = sam_train.prepare_model(args.model_name, args.dataset, int(args.seed)).to(device)
        benign_res = train_sgd(
            model=benign_model,
            train_ds=train_aug_ds,
            eval_ds=train_eval_ds,
            device=device,
            epochs=int(args.epochs),
            lr=float(args.lr),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.train_batch_size),
            num_workers=int(args.train_num_workers),
            stop_train_acc=float(args.stop_train_acc),
            log_prefix="[benign]",
            lr_schedule=str(args.lr_schedule),
        )
        if benign_ckpt is not None:
            save_train_result(benign_ckpt, benign_res)
            print(f"[benign] saved ckpt to {benign_ckpt}", flush=True)
    benign_test = evaluate_test_acc(benign_res.model, test_ds, device=device, batch_size=int(args.eval_batch_size))
    print(f"[benign] test_acc={benign_test['test_acc']:.4f}", flush=True)

    faulty_key = make_cache_key("faulty", {**common_fields, "noise": args.noise_prob})
    faulty_ckpt = (cache_dir / f"{faulty_key}.pt") if cache_dir is not None else None
    if faulty_ckpt is not None and faulty_ckpt.exists():
        print(f"[faulty] loading cached ckpt from {faulty_ckpt}", flush=True)
        faulty_template = sam_train.prepare_model(args.model_name, args.dataset, int(args.seed))
        faulty_res = load_train_result(faulty_ckpt, faulty_template, device=device)
    else:
        sam_train.set_seed(int(args.seed))
        faulty_model = sam_train.prepare_model(args.model_name, args.dataset, int(args.seed)).to(device)
        faulty_res = train_sgd(
            model=faulty_model,
            train_ds=noisy_train_aug_ds,
            eval_ds=noisy_train_eval_ds,
            device=device,
            epochs=int(args.epochs),
            lr=float(args.lr),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.train_batch_size),
            num_workers=int(args.train_num_workers),
            stop_train_acc=float(args.stop_train_acc),
            log_prefix="[faulty]",
            lr_schedule=str(args.lr_schedule),
        )
        if faulty_ckpt is not None:
            save_train_result(faulty_ckpt, faulty_res)
            print(f"[faulty] saved ckpt to {faulty_ckpt}", flush=True)
    faulty_test = evaluate_test_acc(faulty_res.model, test_ds, device=device, batch_size=int(args.eval_batch_size))
    print(f"[faulty] test_acc={faulty_test['test_acc']:.4f}", flush=True)

    # One-shot full-dataset ASAM sharpness on the clean training data.
    print("[noisy_pair] computing static (full-batch ASAM) sharpness on clean train data...", flush=True)
    benign_static = static_sharpness_once(
        benign_res.model, train_eval_ds, device=device,
        batch_size=int(args.eval_batch_size), num_workers=int(args.train_num_workers),
    )
    faulty_static = static_sharpness_once(
        faulty_res.model, train_eval_ds, device=device,
        batch_size=int(args.eval_batch_size), num_workers=int(args.train_num_workers),
    )
    print(f"[benign] static_sharpness={benign_static:.6g}  [faulty] static_sharpness={faulty_static:.6g}", flush=True)

    sweeps = build_sweeps(args)
    print(f"[noisy_pair] running {len(sweeps)} probe configuration(s)", flush=True)

    sweep_results = []
    for sweep in sweeps:
        sid = sweep.short_id()
        print(f"\n[noisy_pair] sweep {sid}", flush=True)
        probe_cfg = sweep.to_probe_args()
        benign_probe = probe_and_score(
            model=benign_res.model,
            train_ds=train_eval_ds,
            test_ds=test_ds,
            out_dir=out_dir / "sweeps" / sid / "benign",
            run_id=f"benign__{sid}",
            device=device,
            meta={"model_type": "benign", "noise_prob": 0.0, "sweep_id": sid, **benign_test,
                  "train_acc": benign_res.train_acc, "train_loss": benign_res.train_loss},
            training_lr=benign_res.initial_lr,
            training_wd=benign_res.initial_wd,
            continuing_lr=benign_res.continuing_lr,
            continuing_wd=benign_res.continuing_wd,
            probe=probe_cfg,
        )
        faulty_probe = probe_and_score(
            model=faulty_res.model,
            train_ds=train_eval_ds,
            test_ds=test_ds,
            out_dir=out_dir / "sweeps" / sid / "faulty",
            run_id=f"faulty__{sid}",
            device=device,
            meta={"model_type": "faulty", "noise_prob": float(args.noise_prob), "sweep_id": sid, **faulty_test,
                  "train_acc": faulty_res.train_acc, "train_loss": faulty_res.train_loss},
            training_lr=faulty_res.initial_lr,
            training_wd=faulty_res.initial_wd,
            continuing_lr=faulty_res.continuing_lr,
            continuing_wd=faulty_res.continuing_wd,
            probe=probe_cfg,
        )
        sweep_results.append((sweep, benign_probe, faulty_probe))

    df_scores_sel = parse_df_scores(args.df_scores)
    all_df = all_sweeps_table(sweep_results, score_columns=df_scores_sel)
    all_df.to_csv(out_dir / "all_sweeps.csv", index=False)
    best_df = best_per_score(sweep_results, score_columns=df_scores_sel)
    best_df.to_csv(out_dir / "best_per_score.csv", index=False)

    # Pair summary uses the first sweep.
    first_b, first_f = sweep_results[0][1], sweep_results[0][2]
    benign_row = {**benign_test, "train_acc": benign_res.train_acc, "train_loss": benign_res.train_loss, **first_b}
    faulty_row = {**faulty_test, "train_acc": faulty_res.train_acc, "train_loss": faulty_res.train_loss, **first_f}
    summary = summarize_pair(benign=benign_row, faulty=faulty_row)
    summary.to_csv(out_dir / "summary.csv", index=False)

    static_df = static_comparison_table(
        observed_acc_benign=float(benign_test["test_acc"]),
        observed_acc_faulty=float(faulty_test["test_acc"]),
        static_benign=float(benign_static),
        static_faulty=float(faulty_static),
    )
    static_df.to_csv(out_dir / "static_comparison.csv", index=False)
    print("\n[noisy_pair] static benign/faulty ratios :\n"
          + static_df.to_string(index=False), flush=True)

    print("\n[noisy_pair] best probe config per DF score (min benign/faulty):\n"
          + best_df[["score", "best_ratio", "benign", "faulty", "sweep_id"]].to_string(index=False), flush=True)
    print(f"\n[noisy_pair] artifacts -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
