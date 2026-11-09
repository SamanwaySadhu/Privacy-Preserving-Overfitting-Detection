#!/usr/bin/env python3
"""Post-quantization pair: SAM-trained benign vs SGD-trained faulty, probed pre-quantization.

Reports both pre- and post-quantization test accuracy alongside DF scores so the
sharpness signal can be compared against the actual quantization-induced drop.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

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
    train_sam,
    train_sgd,
)

DFLAT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(DFLAT_ROOT / "df_utils"))
import sam_train  # noqa: E402


@torch.no_grad()
def quantize_model_inplace(model: nn.Module, *, n_bits: int) -> None:
    """Symmetric per-tensor `n_bits` weight quantization, in-place, dequantized back to float."""
    bits = int(n_bits)
    if bits <= 0:
        raise ValueError("n_bits must be > 0")
    qmax = float(2 ** (bits - 1) - 1)
    for p in model.parameters():
        if not p.is_floating_point():
            continue
        scale = float(p.detach().abs().max().item())
        if scale <= 0 or not (scale == scale):
            continue
        step = scale / qmax
        q = torch.round(p / step).clamp_(-qmax, qmax)
        p.copy_(q * step)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=DFLAT_ROOT / "data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", type=str, default="vgg16_bn")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--train-num-workers", type=int, default=2)
    p.add_argument("--stop-train-acc", type=float, default=0.98, help="0 disables early stopping)")
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--sgd-lr", type=float, default=0.1, help="SGD learning rate (SAM uses --lr)")
    p.add_argument("--sam-train-rho", type=float, default=0.05, help="SAM rho used in the benign training (not the probe)")
    p.add_argument("--sam-train-adaptive", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--quant-bits", type=int, default=4)
    add_probe_sweep_args(p)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--ckpt-cache-dir", type=Path, default=DFLAT_ROOT / "artifacts" / "ckpts" / "quantization_pair",
                   help="Directory to cache trained-from-scratch checkpoints. Re-runs with matching args load from here.")
    p.add_argument("--ckpt-cache", action=argparse.BooleanOptionalAction, default=True,
                   help="If false, ignore the cache and always retrain.")
    args = p.parse_args(argv)

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = DFLAT_ROOT / "artifacts" / f"quantization_pair_{stamp}"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    sam_train.set_seed(int(args.seed))
    device = resolve_device(prefer_index=int(args.device_index), allow_cpu_fallback=True)
    print(f"[quantization_pair] device={device}", flush=True)

    train_aug_ds, test_ds = sam_train.prepare_dataset(
        args.dataset, root=str(args.data_root), normalize=True, randomize=True, cutout=False,
    )
    train_eval_ds, _ = sam_train.prepare_dataset(
        args.dataset, root=str(args.data_root), normalize=True, randomize=False, cutout=False,
    )

    cache_dir = Path(args.ckpt_cache_dir).expanduser().resolve() if args.ckpt_cache else None
    common_fields = {
        "model": args.model_name, "ds": args.dataset, "seed": args.seed,
        "lr": args.lr, "mom": args.momentum, "wd": args.weight_decay,
        "bs": args.train_batch_size, "ep": args.epochs, "stop": args.stop_train_acc,
    }

    benign_key = make_cache_key("benign_sam", {
        **common_fields, "rho": args.sam_train_rho, "ada": args.sam_train_adaptive,
    })
    benign_ckpt = (cache_dir / f"{benign_key}.pt") if cache_dir is not None else None
    if benign_ckpt is not None and benign_ckpt.exists():
        print(f"[benign-sam] loading cached ckpt from {benign_ckpt}", flush=True)
        benign_template = sam_train.prepare_model(args.model_name, args.dataset, int(args.seed))
        benign_res = load_train_result(benign_ckpt, benign_template, device=device)
    else:
        sam_train.set_seed(int(args.seed))
        benign_model = sam_train.prepare_model(args.model_name, args.dataset, int(args.seed)).to(device)
        benign_res = train_sam(
            model=benign_model,
            train_ds=train_aug_ds,
            eval_ds=train_eval_ds,
            device=device,
            epochs=int(args.epochs),
            lr=float(args.lr),
            rho=float(args.sam_train_rho),
            adaptive=bool(args.sam_train_adaptive),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.train_batch_size),
            num_workers=int(args.train_num_workers),
            stop_train_acc=float(args.stop_train_acc),
            log_prefix="[benign-sam]",
        )
        if benign_ckpt is not None:
            save_train_result(benign_ckpt, benign_res)
            print(f"[benign-sam] saved ckpt to {benign_ckpt}", flush=True)
    benign_pre = evaluate_test_acc(benign_res.model, test_ds, device=device, batch_size=int(args.eval_batch_size))
    print(f"[benign-sam] pre-quant test_acc={benign_pre['test_acc']:.4f}", flush=True)

    faulty_key = make_cache_key("faulty_sgd", common_fields)
    faulty_ckpt = (cache_dir / f"{faulty_key}.pt") if cache_dir is not None else None
    if faulty_ckpt is not None and faulty_ckpt.exists():
        print(f"[faulty-sgd] loading cached ckpt from {faulty_ckpt}", flush=True)
        faulty_template = sam_train.prepare_model(args.model_name, args.dataset, int(args.seed))
        faulty_res = load_train_result(faulty_ckpt, faulty_template, device=device)
    else:
        sam_train.set_seed(int(args.seed))
        faulty_model = sam_train.prepare_model(args.model_name, args.dataset, int(args.seed)).to(device)
        faulty_res = train_sgd(
            model=faulty_model,
            train_ds=train_aug_ds,
            eval_ds=train_eval_ds,
            device=device,
            epochs=int(args.epochs),
            lr=float(args.sgd_lr),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.train_batch_size),
            num_workers=int(args.train_num_workers),
            stop_train_acc=float(args.stop_train_acc),
            log_prefix="[faulty-sgd]",
        )
        if faulty_ckpt is not None:
            save_train_result(faulty_ckpt, faulty_res)
            print(f"[faulty-sgd] saved ckpt to {faulty_ckpt}", flush=True)
    faulty_pre = evaluate_test_acc(faulty_res.model, test_ds, device=device, batch_size=int(args.eval_batch_size))
    print(f"[faulty-sgd] pre-quant test_acc={faulty_pre['test_acc']:.4f}", flush=True)

    benign_q = copy.deepcopy(benign_res.model).to(device)
    faulty_q = copy.deepcopy(faulty_res.model).to(device)
    quantize_model_inplace(benign_q, n_bits=int(args.quant_bits))
    quantize_model_inplace(faulty_q, n_bits=int(args.quant_bits))
    benign_post = evaluate_test_acc(benign_q, test_ds, device=device, batch_size=int(args.eval_batch_size))
    faulty_post = evaluate_test_acc(faulty_q, test_ds, device=device, batch_size=int(args.eval_batch_size))
    print(f"[benign-sam] post-quant test_acc={benign_post['test_acc']:.4f}", flush=True)
    print(f"[faulty-sgd] post-quant test_acc={faulty_post['test_acc']:.4f}", flush=True)


    print("[quantization_pair] computing static (full-batch ASAM) sharpness on clean train data...", flush=True)
    benign_static = static_sharpness_once(
        benign_res.model, train_eval_ds, device=device,
        batch_size=int(args.eval_batch_size), num_workers=int(args.train_num_workers),
    )
    faulty_static = static_sharpness_once(
        faulty_res.model, train_eval_ds, device=device,
        batch_size=int(args.eval_batch_size), num_workers=int(args.train_num_workers),
    )
    print(f"[benign-sam] static_sharpness={benign_static:.6g}  [faulty-sgd] static_sharpness={faulty_static:.6g}", flush=True)

    sweeps = build_sweeps(args)
    print(f"[quantization_pair] running {len(sweeps)} probe configuration(s)", flush=True)

    sweep_results = []
    for sweep in sweeps:
        sid = sweep.short_id()
        print(f"\n[quantization_pair] sweep {sid}", flush=True)
        probe_cfg = sweep.to_probe_args()
        benign_probe = probe_and_score(
            model=benign_res.model,
            train_ds=train_eval_ds,
            test_ds=test_ds,
            out_dir=out_dir / "sweeps" / sid / "benign",
            run_id=f"benign__{sid}",
            device=device,
            meta={
                "model_type": "benign", "trainer": "sam", "sweep_id": sid,
                "pre_quant_test_acc": benign_pre["test_acc"],
                "post_quant_test_acc": benign_post["test_acc"],
                "train_acc": benign_res.train_acc, "train_loss": benign_res.train_loss,
            },
            training_lr=benign_res.initial_lr, training_wd=benign_res.initial_wd,
            continuing_lr=benign_res.continuing_lr, continuing_wd=benign_res.continuing_wd,
            probe=probe_cfg,
        )
        faulty_probe = probe_and_score(
            model=faulty_res.model,
            train_ds=train_eval_ds,
            test_ds=test_ds,
            out_dir=out_dir / "sweeps" / sid / "faulty",
            run_id=f"faulty__{sid}",
            device=device,
            meta={
                "model_type": "faulty", "trainer": "sgd", "sweep_id": sid,
                "pre_quant_test_acc": faulty_pre["test_acc"],
                "post_quant_test_acc": faulty_post["test_acc"],
                "train_acc": faulty_res.train_acc, "train_loss": faulty_res.train_loss,
            },
            training_lr=faulty_res.initial_lr, training_wd=faulty_res.initial_wd,
            continuing_lr=faulty_res.continuing_lr, continuing_wd=faulty_res.continuing_wd,
            probe=probe_cfg,
        )
        sweep_results.append((sweep, benign_probe, faulty_probe))

    df_scores_sel = parse_df_scores(args.df_scores)
    all_df = all_sweeps_table(sweep_results, score_columns=df_scores_sel)
    all_df.to_csv(out_dir / "all_sweeps.csv", index=False)
    best_df = best_per_score(sweep_results, score_columns=df_scores_sel)
    best_df.to_csv(out_dir / "best_per_score.csv", index=False)

    first_b, first_f = sweep_results[0][1], sweep_results[0][2]
    benign_row: Dict[str, float] = {
        "pre_quant_test_acc": benign_pre["test_acc"],
        "post_quant_test_acc": benign_post["test_acc"],
        "train_acc": benign_res.train_acc, "train_loss": benign_res.train_loss,
        **first_b,
    }
    faulty_row: Dict[str, float] = {
        "pre_quant_test_acc": faulty_pre["test_acc"],
        "post_quant_test_acc": faulty_post["test_acc"],
        "train_acc": faulty_res.train_acc, "train_loss": faulty_res.train_loss,
        **first_f,
    }
    summary = summarize_pair(benign=benign_row, faulty=faulty_row)
    summary.to_csv(out_dir / "summary.csv", index=False)

    static_df = static_comparison_table(
        observed_acc_benign=float(benign_pre["test_acc"]),
        observed_acc_faulty=float(faulty_pre["test_acc"]),
        static_benign=float(benign_static),
        static_faulty=float(faulty_static),
    )
    static_df.to_csv(out_dir / "static_comparison.csv", index=False)
    print("\n[quantization_pair] static benign/faulty ratios (one-shot, sweep-independent;"
          " observed_acc = pre-quant test acc):\n"
          + static_df.to_string(index=False), flush=True)

    print("\n[quantization_pair] best probe config per DF score (min benign/faulty):\n"
          + best_df[["score", "best_ratio", "benign", "faulty", "sweep_id"]].to_string(index=False), flush=True)
    print(f"\n[quantization_pair] artifacts -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
