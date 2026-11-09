#!/usr/bin/env python3
"""probe a provided clean and backdoor checkpoint
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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
    parse_df_scores,
    probe_and_score,
    resolve_device,
    static_comparison_table,
    static_sharpness_once,
    summarize_pair,
)

DFLAT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(DFLAT_ROOT / "df_utils"))
import sam_train  # noqa: E402


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if isinstance(first_key, str) and first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "net", "weights", "model_state", "ema_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
    raise RuntimeError(f"Unrecognized checkpoint format: {type(payload)}")


def _load_backdoorbox_resnet(backdoorbox_root: Path, depth: int = 18, num_classes: int = 10) -> nn.Module:
    """Import BackdoorBox's `core.models.resnet.ResNet(depth)` from a local checkout."""
    root = Path(backdoorbox_root).expanduser().resolve()
    rn_path = root / "core" / "models" / "resnet.py"
    if not rn_path.exists():
        raise FileNotFoundError(f"BackdoorBox ResNet not found at {rn_path}")

    if "core" not in sys.modules:
        sys.modules["core"] = types.ModuleType("core")
        sys.modules["core"].__path__ = [str(root / "core")]
    if "core.models" not in sys.modules:
        sys.modules["core.models"] = types.ModuleType("core.models")
        sys.modules["core.models"].__path__ = [str(root / "core" / "models")]

    spec = importlib.util.spec_from_file_location("core.models.resnet", str(rn_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ResNet(int(depth), num_classes=int(num_classes))


def _build_model(args: argparse.Namespace) -> nn.Module:
    if args.backdoorbox_root:
        depth = int(args.backdoorbox_resnet_depth)
        num_classes = 100 if args.dataset == "cifar100" else 10
        return _load_backdoorbox_resnet(Path(args.backdoorbox_root), depth=depth, num_classes=num_classes)
    return sam_train.prepare_model(args.model_name, args.dataset, int(args.seed))


def _load_checkpoint(path: Path, template: nn.Module) -> nn.Module:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = _strip_module_prefix(_extract_state_dict(payload))
    missing, unexpected = template.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[probe_backdoor_pair] non-strict load: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    return template


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=DFLAT_ROOT / "data")
    p.add_argument("--clean-ckpt", type=Path, required=True)
    p.add_argument("--backdoor-ckpt", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-name", type=str, default="vgg16_bn", help="df_utils/sam_train arch name (ignored if --backdoorbox-root is set)")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--backdoorbox-root", type=Path, default=None, help="Path to BackdoorBox install. If set, use core.models.resnet.ResNet")
    p.add_argument("--backdoorbox-resnet-depth", type=int, default=18)
    p.add_argument(
        "--input-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If False, drop CIFAR mean/std normalization (use ToTensor only). BackdoorBox-trained models expect this.",
    )
    p.add_argument("--eval-batch-size", type=int, default=256)
    add_probe_sweep_args(p)
    p.add_argument("--device-index", type=int, default=0)
    args = p.parse_args(argv)

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = DFLAT_ROOT / "artifacts" / f"backdoor_pair_{stamp}"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    sam_train.set_seed(int(args.seed))
    device = resolve_device(prefer_index=int(args.device_index), allow_cpu_fallback=True)
    print(f"[backdoor_pair] device={device}", flush=True)

    train_eval_ds, test_ds = sam_train.prepare_dataset(
        args.dataset, root=str(args.data_root),
        normalize=bool(args.input_normalize), randomize=False, cutout=False,
    )

    clean_model = _load_checkpoint(Path(args.clean_ckpt), _build_model(args)).to(device)
    backdoor_model = _load_checkpoint(Path(args.backdoor_ckpt), _build_model(args)).to(device)

    clean_test = evaluate_test_acc(clean_model, test_ds, device=device, batch_size=int(args.eval_batch_size))
    backdoor_test = evaluate_test_acc(backdoor_model, test_ds, device=device, batch_size=int(args.eval_batch_size))
    print(f"[clean]    test_acc={clean_test['test_acc']:.4f}", flush=True)
    print(f"[backdoor] test_acc={backdoor_test['test_acc']:.4f}", flush=True)

    # One-shot full-dataset ASAM sharpness on the clean training data.
    print("[backdoor_pair] computing static (full-batch ASAM) sharpness on clean train data...", flush=True)
    clean_static = static_sharpness_once(
        clean_model, train_eval_ds, device=device,
        batch_size=int(args.eval_batch_size), num_workers=0,
    )
    backdoor_static = static_sharpness_once(
        backdoor_model, train_eval_ds, device=device,
        batch_size=int(args.eval_batch_size), num_workers=0,
    )
    print(f"[clean]    static_sharpness={clean_static:.6g}  [backdoor] static_sharpness={backdoor_static:.6g}", flush=True)

    sweeps = build_sweeps(args)
    print(f"[backdoor_pair] running {len(sweeps)} probe configuration(s)", flush=True)

    sweep_results = []
    for sweep in sweeps:
        sid = sweep.short_id()
        print(f"\n[backdoor_pair] sweep {sid}", flush=True)
        probe_cfg = sweep.to_probe_args()
        clean_probe = probe_and_score(
            model=clean_model,
            train_ds=train_eval_ds,
            test_ds=test_ds,
            out_dir=out_dir / "sweeps" / sid / "clean",
            run_id=f"clean__{sid}",
            device=device,
            meta={"model_type": "clean", "sweep_id": sid, **clean_test, "ckpt_path": str(args.clean_ckpt)},
            training_lr=0.1, training_wd=5e-4,
            continuing_lr=0.1, continuing_wd=5e-4,
            probe=probe_cfg,
        )
        backdoor_probe = probe_and_score(
            model=backdoor_model,
            train_ds=train_eval_ds,
            test_ds=test_ds,
            out_dir=out_dir / "sweeps" / sid / "backdoor",
            run_id=f"backdoor__{sid}",
            device=device,
            meta={"model_type": "backdoor", "sweep_id": sid, **backdoor_test, "ckpt_path": str(args.backdoor_ckpt)},
            training_lr=0.1, training_wd=5e-4,
            continuing_lr=0.1, continuing_wd=5e-4,
            probe=probe_cfg,
        )
        sweep_results.append((sweep, clean_probe, backdoor_probe))

    df_scores_sel = parse_df_scores(args.df_scores)
    all_df = all_sweeps_table(sweep_results, score_columns=df_scores_sel)
    all_df = all_df.rename(columns={"benign": "clean", "faulty": "backdoor",
                                    "ratio_benign_over_faulty": "ratio_clean_over_backdoor"})
    all_df.to_csv(out_dir / "all_sweeps.csv", index=False)
    best_df = best_per_score(sweep_results, score_columns=df_scores_sel)
    best_df = best_df.rename(columns={"benign": "clean", "faulty": "backdoor"})
    best_df.to_csv(out_dir / "best_per_score.csv", index=False)

    first_b, first_f = sweep_results[0][1], sweep_results[0][2]
    benign_row = {**clean_test, **first_b}
    faulty_row = {**backdoor_test, **first_f}
    summary = summarize_pair(benign=benign_row, faulty=faulty_row)
    summary = summary.rename(columns={"benign": "clean", "faulty": "backdoor",
                                      "diff_faulty_minus_benign": "diff_backdoor_minus_clean"})
    summary.to_csv(out_dir / "summary.csv", index=False)

    static_df = static_comparison_table(
        observed_acc_benign=float(clean_test["test_acc"]),
        observed_acc_faulty=float(backdoor_test["test_acc"]),
        static_benign=float(clean_static),
        static_faulty=float(backdoor_static),
        benign_label="clean", faulty_label="backdoor",
    )
    static_df = static_df.rename(columns={"ratio_benign_over_faulty": "ratio_clean_over_backdoor"})
    static_df.to_csv(out_dir / "static_comparison.csv", index=False)
    print("\n[backdoor_pair] static clean/backdoor ratios :\n"
          + static_df.to_string(index=False), flush=True)

    print("\n[backdoor_pair] best probe config per DF score (min clean/backdoor):\n"
          + best_df[["score", "best_ratio", "clean", "backdoor", "sweep_id"]].to_string(index=False), flush=True)
    print(f"\n[backdoor_pair] artifacts -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
