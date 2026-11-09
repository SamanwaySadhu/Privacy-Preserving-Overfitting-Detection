#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from probe_common import *  # noqa: F401,F403  (generic probe/score helpers)

# ----------------------------- Repo/module loading -----------------------------


def _guess_default_sharp_agent_root() -> Path:
    env = str(os.environ.get("SHARP_AGENT_ROOT", "")).strip()
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    here = Path(__file__).resolve()
    user_root = here.parents[2]
    candidates.append(user_root / "Sharp-Agent")
    candidates.append(user_root / "SAPA" / "Sharp-Agent")
    for candidate in candidates:
        if (candidate / "sleeper_agent.py").exists() and (candidate / "forest").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the Sharp-Agent repo root (expects a dir containing "
        "sleeper_agent.py and forest/). Clone https://github.com/PengfeiHePower/SAPA "
        "and pass --sharp-agent-root /path/to/SAPA/Sharp-Agent or set SHARP_AGENT_ROOT."
    )


def _resolve_path_arg(
    raw_value: Optional[str],
    *,
    anchor_root: Path,
    prefer_existing: bool,
    allow_empty: bool = False,
) -> Optional[str]:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return "" if allow_empty else text

    p = Path(text).expanduser()
    if p.is_absolute():
        return str(p.resolve())

    cwd_candidate = (Path.cwd() / p).resolve()
    anchor_candidate = (anchor_root / p).resolve()

    if prefer_existing:
        if cwd_candidate.exists():
            return str(cwd_candidate)
        if anchor_candidate.exists():
            return str(anchor_candidate)
        return str(anchor_candidate)

    if cwd_candidate.exists() and not anchor_candidate.exists():
        return str(cwd_candidate)
    return str(anchor_candidate)


def _normalize_sapa_runtime_paths(args: argparse.Namespace, *, sharp_agent_root: Path) -> None:
    output_like_keys = ("table_path", "poison_path", "modelsave_path")
    input_or_mixed_keys = ("data_path", "lmdb_path", "benchmark")

    for key in output_like_keys:
        if hasattr(args, key):
            value = getattr(args, key, None)
            resolved = _resolve_path_arg(
                value,
                anchor_root=sharp_agent_root,
                prefer_existing=False,
                allow_empty=False,
            )
            setattr(args, key, resolved)

    for key in input_or_mixed_keys:
        if hasattr(args, key):
            value = getattr(args, key, None)
            resolved = _resolve_path_arg(
                value,
                anchor_root=sharp_agent_root,
                prefer_existing=True,
                allow_empty=False,
            )
            setattr(args, key, resolved)

    if hasattr(args, "load_patch"):
        raw_patch = getattr(args, "load_patch", None)
        patch_value = str(raw_patch).strip() if raw_patch is not None else ""
        if patch_value == "":
            return
        resolved_patch = _resolve_path_arg(
            patch_value,
            anchor_root=sharp_agent_root,
            prefer_existing=True,
            allow_empty=True,
        )
        setattr(args, "load_patch", resolved_patch)
        if resolved_patch and not Path(resolved_patch).exists():
            raise FileNotFoundError(
                f"Patch file not found: {resolved_patch}. "
                f"Original value='{patch_value}'. "
                "Provide an absolute --load_patch path or place the file under --sharp-agent-root."
            )


# ----------------------------- Pair utilities -----------------------------


def apply_pair_keys(args: argparse.Namespace, pair_idx: int) -> argparse.Namespace:
    out = copy.deepcopy(args)
    if bool(args.pair_override_keys):
        seed = int(args.pair_seed_base) + int(pair_idx) * int(args.pair_seed_stride)
        out.modelkey = int(seed)
        out.poisonkey = str(int(seed))
        return out

    # Offset explicit numeric keys for reproducible multi-pair runs when possible.
    if int(args.pair_count) > 1:
        if out.modelkey is not None:
            out.modelkey = int(out.modelkey) + int(pair_idx) * int(args.pair_seed_stride)
        if out.poisonkey is not None:
            key_text = str(out.poisonkey).strip()
            if key_text.isdigit():
                out.poisonkey = str(int(key_text) + int(pair_idx) * int(args.pair_seed_stride))
            elif int(pair_idx) > 0:
                raise ValueError(
                    "poisonkey is non-numeric (deterministic triplet style), cannot auto-offset for pair_idx>0. "
                    "Use --pair-override-keys."
                )
    return out



# ----------------------------- Main -----------------------------


def build_parser(forest_module, sharp_agent_root: Path) -> argparse.ArgumentParser:
    parser = forest_module.options()

    # Recommended sleeper_agent defaults from Sharp-Agent README.
    parser.set_defaults(
        net=["ResNet18"],
        dataset="CIFAR10",
        patch_size=8,
        budget=0.01,
        eps=16.0,
        pbatch=128,
        epochs=80,
        sources=1000,
        source_gradient_batch=300,
        save="numpy",
        optimization="standard2",
        source_criterion="sharpness",
        sharpsigma=0.01,
        name="sapa_resnet18_cifar10",
    )

    # Repos/outputs.
    parser.add_argument("--sharp-agent-root", type=Path, default=sharp_agent_root)
    parser.add_argument("--dflat-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=None)

    # Pair control.
    parser.add_argument("--pair-count", type=int, default=1)
    parser.add_argument("--pair-seed-base", type=int, default=0)
    parser.add_argument("--pair-seed-stride", type=int, default=1)
    parser.add_argument(
        "--pair-override-keys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, overwrite modelkey/poisonkey per pair using pair-seed-base/stride.",
    )
    parser.add_argument(
        "--backdoor-train-max-epoch",
        type=int,
        default=0,
        help="If >0, use this max_epoch for backdoor retraining. If 0, use model defs.epochs.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run clean/backdoor training only; skip directional probing and score aggregation.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Skip all training and probe checkpoints from --probe-pair-dir.",
    )
    parser.add_argument(
        "--poison-training-only",
        action="store_true",
        help=(
            "Skip clean training and poison crafting. Load clean_model.pt + poison_delta.pt from "
            "--probe-pair-dir, train only the backdoor model, then continue eval/probing."
        ),
    )
    parser.add_argument(
        "--probe-pair-dir",
        type=Path,
        default=None,
        help=(
            "Path to an existing pair directory (e.g., .../pair_000). "
            "For --probe-only: needs clean_model.pt + backdoor_model.pt. "
            "For --poison-training-only: needs clean_model.pt + poison_delta.pt."
        ),
    )

    # Evaluation/full-batch settings.
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--fullbatch-batch-size", type=int, default=256)
    parser.add_argument(
        "--fullbatch-adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, use adaptive term (w * grad) for full-batch norm like adaptive SAM.",
    )

    # Probe settings
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="Comma-separated physical GPU ids, e.g. '0,3'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Optional worker cap for probe orchestration compatibility.",
    )
    parser.add_argument(
        "--probe-device-index",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-allow-cpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If CUDA is unavailable, patch df_utils probing to run on CPU instead of exiting.",
    )

    parser.add_argument("--sweep-probe-steps", type=_parse_csv_ints, default=[10])
    parser.add_argument("--sweep-probe-epochs", type=_parse_csv_ints, default=[0])
    parser.add_argument("--sweep-probe-extra-steps", type=_parse_csv_ints, default=[0])
    parser.add_argument("--sweep-record-every-steps", type=_parse_csv_ints, default=[0])
    parser.add_argument("--sweep-record-points", type=_parse_csv_ints, default=[10])
    parser.add_argument("--sweep-probe-step-batch-size", type=_parse_csv_ints, default=[8])
    parser.add_argument("--sweep-step-train-eval-size", type=_parse_csv_ints, default=[64])
    parser.add_argument("--sweep-sam-step-batch-size", type=_parse_csv_ints, default=[16])
    parser.add_argument("--sweep-sam-rho", type=_parse_csv_floats, default=[0.05])
    parser.add_argument("--sweep-sam-lr", type=_parse_csv_floats, default=[1e-6])
    parser.add_argument("--sweep-sam-base-opt", type=_parse_csv_strings, default=["adamw"])

    parser.add_argument("--sam-weight-decay", type=float, default=0.1)
    parser.add_argument("--sam-momentum", type=float, default=0.9)
    parser.add_argument("--sam-adaptive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-params-source",
        type=str,
        default="continuing",
        choices=["specified", "initial", "continuing"],
    )
    parser.add_argument("--flat-r", type=float, default=1.0)
    parser.add_argument("--flat-delta", type=float, default=1.0)
    parser.add_argument("--flat-norm", type=str, default="l2")
    parser.add_argument("--include-full-batch-step", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-unfreeze", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--auto-microbatch-size", type=int, default=64)
    parser.add_argument("--max-probe-sweeps", type=int, default=0)

    # Directional scoring settings.
    parser.add_argument("--directional-flat-col", type=str, default="flat_primal_freeze")
    parser.add_argument("--min-step-index", type=int, default=0)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--score-min-step", dest="min_step_index", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--score-min-steps", dest="min_steps", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    # Trend settings.
    parser.add_argument("--trend-min-step", type=int, default=0)
    parser.add_argument("--trend-min-points", type=int, default=3)
    parser.add_argument("--trend-window-sizes", type=str, default=None)
    parser.add_argument("--trend-window-trim-fractions", type=str, default=None)
    parser.add_argument("--trend-slope-smoother-params", type=str, default=None)
    parser.add_argument("--trend-cumulative-thresholds", type=str, default=None)
    parser.add_argument("--trend-sign-thresholds", type=str, default=None)
    parser.add_argument("--trend-monotone-smoother-params", type=str, default=None)
    parser.add_argument(
        "--trend-auto-scale-from-steps",
        type=str,
        default="false",
        choices=["true", "false"],
    )

    # Output behavior.
    parser.add_argument("--summary-top-k", type=int, default=12)
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-poison-delta",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--export-poison",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, call data.export_poison(poison_delta, mode=args.save).",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--sharp-agent-root", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv_list)

    sharp_agent_root = (
        pre_args.sharp_agent_root.resolve()
        if pre_args.sharp_agent_root is not None
        else _guess_default_sharp_agent_root()
    )
    if str(sharp_agent_root) not in sys.path:
        sys.path.insert(0, str(sharp_agent_root))

    import forest  # noqa: E402

    parser = build_parser(forest, sharp_agent_root=sharp_agent_root)
    args = parser.parse_args(argv_list)

    if isinstance(args.net, str):
        args.net = [args.net]
    if int(args.min_steps) < 2:
        raise ValueError("--min-steps must be >= 2")
    if int(args.min_step_index) < 0:
        raise ValueError("--min-step-index must be >= 0")
    if str(args.flat_norm).strip().lower() not in {"l1", "l2"}:
        raise ValueError(f"Invalid --flat-norm '{args.flat_norm}'. Use 'l1' or 'l2'.")
    if int(args.trend_min_step) < 0:
        raise ValueError("--trend-min-step must be >= 0")
    if int(args.trend_min_points) < 2:
        raise ValueError("--trend-min-points must be >= 2")
    mode_flags = [
        bool(args.train_only),
        bool(args.probe_only),
        bool(args.poison_training_only),
    ]
    if int(sum(1 for flag in mode_flags if flag)) > 1:
        raise ValueError(
            "Only one mode flag may be set among: --train-only, --probe-only, --poison-training-only."
        )

    run_mode = "full"
    if bool(args.train_only):
        run_mode = "train-only"
    elif bool(args.probe_only):
        run_mode = "probe-only"
    elif bool(args.poison_training_only):
        run_mode = "poison-training-only"

    probe_pair_dir: Optional[Path] = None
    if run_mode in {"probe-only", "poison-training-only"}:
        if args.probe_pair_dir is None:
            raise ValueError(f"--{run_mode} requires --probe-pair-dir.")
        probe_pair_dir = Path(args.probe_pair_dir).expanduser().resolve()
        if not probe_pair_dir.exists():
            raise FileNotFoundError(f"--probe-pair-dir does not exist: {probe_pair_dir}")
        if not probe_pair_dir.is_dir():
            raise ValueError(f"--probe-pair-dir must be a directory: {probe_pair_dir}")
        if int(args.pair_count) != 1:
            raise ValueError(f"--{run_mode} currently requires --pair-count 1.")

    sharp_agent_root = Path(args.sharp_agent_root).expanduser().resolve()
    _normalize_sapa_runtime_paths(args, sharp_agent_root=sharp_agent_root)
    dflat_root = Path(args.dflat_root).expanduser().resolve()
    gpu_ids = _parse_csv_ints(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one integer id.")
    if args.probe_device_index is not None:
        args.probe_device_index_effective = int(args.probe_device_index)
    else:
        args.probe_device_index_effective = int(gpu_ids[0])

    if str(sharp_agent_root) not in sys.path:
        sys.path.insert(0, str(sharp_agent_root))

    df_module = load_df_module(dflat_root)
    if not torch.cuda.is_available():
        if bool(args.probe_allow_cpu_fallback):
            def _cpu_set_device(_: int) -> torch.device:
                return torch.device("cpu")
            df_module.set_device = _cpu_set_device
        else:
            raise RuntimeError(
                "CUDA is unavailable and probe_allow_cpu_fallback is disabled; df_utils probing would exit."
            )
    else:
        if int(args.probe_device_index_effective) >= int(torch.cuda.device_count()):
            raise ValueError(
                f"effective probe device index={int(args.probe_device_index_effective)} is out of range for "
                f"{int(torch.cuda.device_count())} visible CUDA device(s)."
            )

    if args.out_dir is None:
        if run_mode in {"probe-only", "poison-training-only"}:
            out_dir = probe_pair_dir.parent
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_dir = dflat_root / "artifacts" / f"sapa_resnet_df_{stamp}"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sweeps: List[ProbeSweep] = []
    if run_mode != "train-only":
        sweeps = build_probe_sweeps(args)
        if int(args.max_probe_sweeps) > 0:
            sweeps = sweeps[: int(args.max_probe_sweeps)]
        if not sweeps:
            raise RuntimeError("No probe sweep configuration resolved.")

    config_payload = {
        "script": str(Path(__file__).resolve()),
        "sharp_agent_root": str(sharp_agent_root),
        "dflat_root": str(dflat_root),
        "out_dir": str(out_dir),
        "run_mode": run_mode,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "num_sweeps": len(sweeps),
        "sweeps": [asdict(s) for s in sweeps],
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    all_rows: List[Dict[str, Any]] = []

    for pair_idx in range(int(args.pair_count)):
        pair_args = apply_pair_keys(args, pair_idx)
        if run_mode in {"probe-only", "poison-training-only"}:
            pair_dir = probe_pair_dir
            pair_row_id = parse_pair_id_from_dir(pair_dir, default=pair_idx)
        else:
            pair_dir = out_dir / f"pair_{pair_idx:03d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            pair_row_id = int(pair_idx)

        if bool(pair_args.deterministic):
            forest.utils.set_deterministic()

        setup = forest.utils.system_startup(pair_args)
        victim = forest.Victim(pair_args, setup=setup)
        data = forest.Kettle(
            pair_args,
            victim.defs.batch_size,
            victim.defs.augmentations,
            victim.defs.mixing_method,
            setup=setup,
        )
        # Keep a fixed clean dataset view for all sharpness/test metrics.
        clean_train_ds_for_metrics = XYDataset(data.trainset)
        clean_val_ds_for_metrics = XYDataset(data.validset)

        if run_mode == "probe-only":
            clean_ckpt = pair_dir / "clean_model.pt"
            backdoor_ckpt = pair_dir / "backdoor_model.pt"

            train_lr_initial = float(getattr(victim.defs, "lr", pair_args.lr if pair_args.lr is not None else 0.1))
            train_wd_initial = float(getattr(victim.defs, "weight_decay", 0.0))

            clean_model = load_checkpoint_model(clean_ckpt, victim.model)
            backdoor_model = load_checkpoint_model(backdoor_ckpt, victim.model)

            # Optimizer state is not saved in model checkpoints; default to defs.
            clean_lr_continuing = float(train_lr_initial)
            clean_wd_continuing = float(train_wd_initial)
            backdoor_lr_continuing = float(train_lr_initial)
            backdoor_wd_continuing = float(train_wd_initial)
        elif run_mode == "poison-training-only":
            witch = forest.Witch(pair_args, setup=setup)
            if bool(pair_args.backdoor_poisoning):
                witch.patch_sources(data)
            if pair_args.pretrained_model:
                raise ValueError("pretrained_model path in this script is not implemented for CIFAR training.")

            clean_ckpt = pair_dir / "clean_model.pt"
            poison_delta_path = pair_dir / "poison_delta.pt"
            if not poison_delta_path.exists():
                raise FileNotFoundError(
                    f"--poison-training-only requires an existing poison_delta.pt at: {poison_delta_path}"
                )

            train_lr_initial = float(getattr(victim.defs, "lr", pair_args.lr if pair_args.lr is not None else 0.1))
            train_wd_initial = float(getattr(victim.defs, "weight_decay", 0.0))
            clean_model = load_checkpoint_model(clean_ckpt, victim.model)
            clean_lr_continuing = float(train_lr_initial)
            clean_wd_continuing = float(train_wd_initial)

            # poison selection is computed from the clean model before brewing.
            victim_state_target = _unwrap_model(victim.model)
            clean_state = clean_model.state_dict()
            missing, unexpected = victim_state_target.load_state_dict(clean_state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"Failed to load clean checkpoint into victim model for poison selection. "
                    f"missing={missing}, unexpected={unexpected}"
                )
            if pair_args.poison_selection_strategy is not None:
                data.select_poisons(victim, pair_args.poison_selection_strategy)

            poison_payload = torch.load(poison_delta_path, map_location="cpu")
            poison_delta = poison_payload["poison_delta"] if isinstance(poison_payload, dict) and "poison_delta" in poison_payload else poison_payload
            if not torch.is_tensor(poison_delta):
                raise ValueError(f"Unsupported poison delta format at {poison_delta_path}: {type(poison_delta)}")
            expected_poison_n = int(len(data.poison_ids))
            loaded_poison_n = int(poison_delta.shape[0]) if poison_delta.ndim > 0 else -1
            if loaded_poison_n != expected_poison_n:
                raise ValueError(
                    "Poison delta shape mismatch after poison selection reconstruction: "
                    f"loaded poison_delta.shape[0]={loaded_poison_n}, expected len(data.poison_ids)={expected_poison_n}. "
                    "Use the same pair seeds/poison-selection settings as the original run, or rerun full mode."
                )

            print(f"[pair {pair_row_id}] training backdoor model from existing poison_delta.pt", flush=True)
            bd_max_epoch = int(args.backdoor_train_max_epoch)
            _ = victim.retrain(data, poison_delta, max_epoch=(None if bd_max_epoch <= 0 else bd_max_epoch))
            backdoor_model = _clone_model_cpu(victim.model)
            backdoor_lr_continuing, backdoor_wd_continuing = _extract_optimizer_lr_wd(
                getattr(victim, "optimizer", None),
                fallback_lr=train_lr_initial,
                fallback_wd=train_wd_initial,
            )

            maybe_save_checkpoint(
                pair_dir / "backdoor_model.pt",
                backdoor_model,
                meta={
                    "pair_id": int(pair_row_id),
                    "model_type": "backdoor",
                    "checkpoint_stage": "post_backdoor_training",
                },
                enabled=bool(args.save_checkpoints),
            )
        else:
            witch = forest.Witch(pair_args, setup=setup)

            if bool(pair_args.backdoor_poisoning):
                witch.patch_sources(data)

            if pair_args.pretrained_model:
                raise ValueError("pretrained_model path in this script is not implemented for CIFAR training.")

            if bool(pair_args.skip_clean_training):
                raise ValueError("skip_clean_training=True is incompatible with this script's clean-vs-backdoor objective.")

            print(f"\n[pair {pair_row_id}] training clean model", flush=True)
            _ = victim.train(data, max_epoch=pair_args.max_epoch)

            clean_model = _clone_model_cpu(victim.model)
            train_lr_initial = float(getattr(victim.defs, "lr", pair_args.lr if pair_args.lr is not None else 0.1))
            train_wd_initial = float(getattr(victim.defs, "weight_decay", 0.0))
            clean_lr_continuing, clean_wd_continuing = _extract_optimizer_lr_wd(
                getattr(victim, "optimizer", None),
                fallback_lr=train_lr_initial,
                fallback_wd=train_wd_initial,
            )

            maybe_save_checkpoint(
                pair_dir / "clean_model.pt",
                clean_model,
                meta={
                    "pair_id": int(pair_row_id),
                    "model_type": "clean",
                    "checkpoint_stage": "post_clean_training",
                },
                enabled=bool(args.save_checkpoints),
            )

            if pair_args.poison_selection_strategy is not None:
                data.select_poisons(victim, pair_args.poison_selection_strategy)

            print(f"[pair {pair_row_id}] brewing poison", flush=True)
            poison_delta = witch.brew(victim, data)

            if bool(args.save_poison_delta):
                torch.save(poison_delta, pair_dir / "poison_delta.pt")

            if bool(args.export_poison) and pair_args.save is not None:
                data.export_poison(poison_delta, path=pair_args.poison_path, mode=pair_args.save)

            print(f"[pair {pair_row_id}] training backdoor model", flush=True)
            bd_max_epoch = int(args.backdoor_train_max_epoch)
            _ = victim.retrain(data, poison_delta, max_epoch=(None if bd_max_epoch <= 0 else bd_max_epoch))
            backdoor_model = _clone_model_cpu(victim.model)
            backdoor_lr_continuing, backdoor_wd_continuing = _extract_optimizer_lr_wd(
                getattr(victim, "optimizer", None),
                fallback_lr=train_lr_initial,
                fallback_wd=train_wd_initial,
            )

            maybe_save_checkpoint(
                pair_dir / "backdoor_model.pt",
                backdoor_model,
                meta={
                    "pair_id": int(pair_row_id),
                    "model_type": "backdoor",
                    "checkpoint_stage": "post_backdoor_training",
                },
                enabled=bool(args.save_checkpoints),
            )

        model_bundles = [
            ("clean", clean_model, clean_lr_continuing, clean_wd_continuing),
            ("backdoor", backdoor_model, backdoor_lr_continuing, backdoor_wd_continuing),
        ]

        for model_type, base_model, model_lr_continuing, model_wd_continuing in model_bundles:
            print(f"[pair {pair_row_id}] evaluating {model_type}", flush=True)
            eval_stats = evaluate_loss_acc(
                base_model,
                clean_val_ds_for_metrics,
                device=setup["device"],
                batch_size=int(args.eval_batch_size),
                num_workers=int(args.eval_num_workers),
            )
            fb_stats = fullbatch_sharpness_once(
                base_model,
                clean_train_ds_for_metrics,
                device=setup["device"],
                batch_size=int(args.fullbatch_batch_size),
                num_workers=int(args.eval_num_workers),
                adaptive=bool(args.fullbatch_adaptive),
            )

            maybe_save_checkpoint(
                pair_dir / f"{model_type}_model.pt",
                base_model,
                meta={
                    "pair_id": pair_row_id,
                    "model_type": model_type,
                    "test_acc": float(eval_stats["test_acc"]),
                    "test_loss": float(eval_stats["test_loss"]),
                    "fullbatch_sharp_l1": float(fb_stats["fullbatch_sharp_l1"]),
                    "fullbatch_sharp_l2": float(fb_stats["fullbatch_sharp_l2"]),
                },
                enabled=bool(args.save_checkpoints),
            )

            if run_mode == "train-only":
                continue

            for sweep in sweeps:
                sweep_id = sweep.sweep_id()
                probe_out_dir = pair_dir / "probe" / model_type / sweep_id
                run_id = f"pair{pair_row_id}_{model_type}_{sweep_id}"

                probe_meta = {
                    "pair_id": int(pair_row_id),
                    "model_type": str(model_type),
                    "dataset": str(pair_args.dataset),
                    "model": str(pair_args.net[0]),
                    "seed": int(pair_args.modelkey if pair_args.modelkey is not None else -1),
                    "optimizer": str(getattr(victim.defs, "optimizer", pair_args.optimization)),
                    "lr": float(sweep.sam_lr),
                    "wd": float(sweep.sam_weight_decay),
                    "batch_size": int(getattr(victim.defs, "batch_size", pair_args.batch_size or 128)),
                    "label_smoothing": 0.0,
                    "dropout": 0.0,
                    "test_acc": float(eval_stats["test_acc"]),
                    "test_loss": float(eval_stats["test_loss"]),
                }

                directional_df = run_probe_and_collect_scores(
                    df_module=df_module,
                    base_model=base_model,
                    train_ds=clean_train_ds_for_metrics,
                    val_ds=clean_val_ds_for_metrics,
                    sweep=sweep,
                    run_id=run_id,
                    out_dir=probe_out_dir,
                    args=args,
                    meta=probe_meta,
                    training_lr=float(train_lr_initial),
                    training_wd=float(train_wd_initial),
                    continuing_lr=float(model_lr_continuing),
                    continuing_wd=float(model_wd_continuing),
                )

                for _, drow in directional_df.iterrows():
                    row: Dict[str, Any] = {
                        "pair_id": int(pair_row_id),
                        "pair_seed": int(pair_args.modelkey if pair_args.modelkey is not None else -1),
                        "model_type": str(model_type),
                        "dataset": str(pair_args.dataset),
                        "model": str(pair_args.net[0]),
                        "sweep_id": str(sweep_id),
                        "flat_norm": str(drow.get("flat_norm", "")),
                        "flat_col": str(drow.get("flat_col", "")),
                        "probe_csv": str(drow.get("probe_csv", "")),
                        "test_loss": float(eval_stats["test_loss"]),
                        "test_acc": float(eval_stats["test_acc"]),
                        "fullbatch_loss": float(fb_stats["fullbatch_loss"]),
                        "fullbatch_acc": float(fb_stats["fullbatch_acc"]),
                        "fullbatch_sharp_l1": float(fb_stats["fullbatch_sharp_l1"]),
                        "fullbatch_sharp_l2": float(fb_stats["fullbatch_sharp_l2"]),
                    }
                    for key, value in asdict(sweep).items():
                        row[key] = value

                    for key in drow.index:
                        if key in {"flat_norm", "flat_col", "probe_csv"}:
                            continue
                        row[key] = drow[key]

                    all_rows.append(row)

    if run_mode == "train-only":
        print("\nTrain-only mode complete.", flush=True)
        print(f"Checkpoints/outputs were written under: {out_dir}", flush=True)
        return 0

    metrics_df = pd.DataFrame(all_rows)
    if metrics_df.empty:
        raise RuntimeError("No metrics rows were produced.")

    metrics_path = out_dir / "pair_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    directional_score_cols = [
        c
        for c in metrics_df.columns
        if c.startswith("FLAT_") or c.startswith("TREND_")
    ]
    metric_columns = [
        c
        for c in [
            "test_acc",
            "test_loss",
            "fullbatch_loss",
            "fullbatch_sharp_l1",
            "fullbatch_sharp_l2",
        ]
        if c in metrics_df.columns
    ] + directional_score_cols

    diff_df, avg_df = build_difference_tables(metrics_df, metric_columns=metric_columns)

    diff_path = out_dir / "pair_metric_differences.csv"
    avg_path = out_dir / "avg_metric_differences.csv"
    diff_df.to_csv(diff_path, index=False)
    avg_df.to_csv(avg_path, index=False)

    summary_text = build_summary_text(
        metrics_df=metrics_df,
        diff_df=diff_df,
        avg_df=avg_df,
        args=args,
        metric_columns=metric_columns,
    )
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    print("\n" + summary_text, flush=True)
    print(f"\nWrote metrics to: {metrics_path}", flush=True)
    print(f"Wrote pair differences to: {diff_path}", flush=True)
    print(f"Wrote averaged differences to: {avg_path}", flush=True)
    print(f"Wrote summary to: {summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

