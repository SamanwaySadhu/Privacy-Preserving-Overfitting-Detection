"""Shared train + probe utilities for the distinguishability experiments."""
from __future__ import annotations

import argparse
import contextlib
import copy
import itertools
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PRECISION_CHOICES = ("fp32", "bf16", "fp16")


def _resolve_amp_dtype(precision: str):
    p = str(precision).strip().lower()
    if p in {"fp32", "float32", "32"}:
        return None
    if p in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if p in {"fp16", "float16", "16"}:
        return torch.float16
    raise ValueError(f"Unsupported precision {precision!r}; expected one of {PRECISION_CHOICES}")


def autocast_ctx(device: torch.device, precision: str = "fp32"):
    dtype = _resolve_amp_dtype(precision)
    if dtype is None:
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type=device.type, dtype=dtype)
    except (RuntimeError, ValueError):
        # Older torch/MPS may not support bf16 autocast.
        return contextlib.nullcontext()


DFLAT_ROOT = Path(__file__).resolve().parents[1]
DF_UTIL = DFLAT_ROOT / "df_utils"
if str(DF_UTIL) not in sys.path:
    sys.path.insert(0, str(DF_UTIL))

import df as df_module  # noqa: E402
import sam_train  # noqa: E402


DEFAULT_SCORE_COLUMNS = list(df_module.SCORE_COLUMNS)


def resolve_device(prefer_index: int = 0, allow_cpu_fallback: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return sam_train.set_device(prefer_index)
    if sam_train.mps_available():
        return torch.device("mps")
    if not allow_cpu_fallback:
        raise RuntimeError("No CUDA or MPS device available.")
    return torch.device("cpu")


def _patch_set_device_for_cpu_if_needed(device: torch.device) -> None:
    # df.run_probe re-resolves the device through df.set_device; force it for non-CUDA backends.
    if device.type == "cpu":
        df_module.set_device = lambda _idx: torch.device("cpu")
    elif device.type == "mps":
        df_module.set_device = lambda _idx: torch.device("mps")


@torch.no_grad()
def evaluate_test_acc(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
    precision: str = "fp32",
) -> Dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    was_training = bool(model.training)
    model.to(device).eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    with autocast_ctx(device, precision):
        for batch in loader:
            xb, yb = batch[0], batch[1]
            xb = xb.to(device, non_blocking=torch.cuda.is_available())
            yb = yb.to(device, non_blocking=torch.cuda.is_available()).long()
            logits = model(xb)
            loss = ce_sum(logits, yb)
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_n += int(yb.numel())
    if was_training:
        model.train()
    denom = max(1, total_n)
    return {
        "test_loss": float(total_loss / denom),
        "test_acc": float(total_correct / denom),
        "test_n": int(total_n),
    }


@torch.no_grad()
def _train_eval_acc(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
    precision: str = "fp32",
) -> Dict[str, float]:
    return evaluate_test_acc(model, dataset, device=device, batch_size=batch_size,
                             num_workers=num_workers, precision=precision)


@dataclass
class TrainResult:
    model: nn.Module
    epochs_trained: int
    train_loss: float
    train_acc: float
    stopped_early: bool
    initial_lr: float
    initial_wd: float
    continuing_lr: float
    continuing_wd: float


def _fmt_cache_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.6g}".replace(".", "p").replace("-", "m").replace("+", "")
    return str(v).replace("/", "_").replace(" ", "_")


def make_cache_key(prefix: str, fields: Dict[str, Any]) -> str:
    """Deterministic filename built from sorted (key, value) pairs."""
    parts = [prefix]
    for k in sorted(fields.keys()):
        parts.append(f"{k}{_fmt_cache_value(fields[k])}")
    return "__".join(parts)


def save_train_result(path: Path, result: TrainResult) -> None:
    """Persist a TrainResult to disk (state_dict + scalar metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": result.model.state_dict(),
        "meta": {
            "epochs_trained": int(result.epochs_trained),
            "train_loss": float(result.train_loss),
            "train_acc": float(result.train_acc),
            "stopped_early": bool(result.stopped_early),
            "initial_lr": float(result.initial_lr),
            "initial_wd": float(result.initial_wd),
            "continuing_lr": float(result.continuing_lr),
            "continuing_wd": float(result.continuing_wd),
        },
    }
    torch.save(payload, str(path))


def load_train_result(path: Path, model: nn.Module, *, device: torch.device) -> TrainResult:
    """Restore a TrainResult previously written with `save_train_result`."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        meta = dict(payload.get("meta", {}))
    else:
        state = payload if isinstance(payload, dict) else {}
        meta = {}
    model.load_state_dict(state, strict=True)
    model.to(device)
    return TrainResult(
        model=model,
        epochs_trained=int(meta.get("epochs_trained", 0)),
        train_loss=float(meta.get("train_loss", float("nan"))),
        train_acc=float(meta.get("train_acc", float("nan"))),
        stopped_early=bool(meta.get("stopped_early", False)),
        initial_lr=float(meta.get("initial_lr", 0.0)),
        initial_wd=float(meta.get("initial_wd", 0.0)),
        continuing_lr=float(meta.get("continuing_lr", 0.0)),
        continuing_wd=float(meta.get("continuing_wd", 0.0)),
    )


LR_SCHEDULE_CHOICES = ("cosine", "multistep", "none")


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, *, kind: str, epochs: int):
    k = str(kind).strip().lower()
    if k in {"none", "off", "constant"}:
        return None
    if k == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    if k == "multistep":
        e = max(1, int(epochs))
        milestones = sorted({max(1, e // 2), max(1, (3 * e) // 4)})
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    raise ValueError(f"Unsupported lr_schedule {kind!r}; expected one of {LR_SCHEDULE_CHOICES}")


def train_sgd(
    *,
    model: nn.Module,
    train_ds: Dataset,
    eval_ds: Dataset,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 128,
    num_workers: int = 2,
    stop_train_acc: float = 0.0,
    log_prefix: str = "[train-sgd]",
    precision: str = "fp32",
    lr_schedule: str = "cosine",
) -> TrainResult:
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    scheduler = _build_lr_scheduler(optimizer, kind=lr_schedule, epochs=int(epochs))
    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )
    last_eval = {"loss": float("nan"), "acc": float("nan")}
    stopped_early = False
    ep = 0
    for ep in range(1, max(1, int(epochs)) + 1):
        model.train()
        # Autocast once per epoch, not per batch.
        with autocast_ctx(device, precision):
            for batch in train_loader:
                xb, yb = batch[0], batch[1]
                xb = xb.to(device, non_blocking=torch.cuda.is_available())
                yb = yb.to(device, non_blocking=torch.cuda.is_available()).long()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                optimizer.step()
        ev = _train_eval_acc(model, eval_ds, device=device, batch_size=512, num_workers=num_workers, precision=precision)
        last_eval = {"loss": ev["test_loss"], "acc": ev["test_acc"]}
        cur_lr = float(optimizer.param_groups[0].get("lr", lr))
        print(f"{log_prefix} ep={ep:03d} lr={cur_lr:.5g} train_loss={last_eval['loss']:.4f} train_acc={last_eval['acc']:.4f}", flush=True)
        if scheduler is not None:
            scheduler.step()
        if float(stop_train_acc) > 0.0 and float(last_eval["acc"]) >= float(stop_train_acc):
            stopped_early = True
            print(f"{log_prefix} early-stop: train_acc>={stop_train_acc:.4f}", flush=True)
            break

    lr_cont = float(optimizer.param_groups[0].get("lr", lr))
    wd_cont = float(optimizer.param_groups[0].get("weight_decay", weight_decay))
    return TrainResult(
        model=model,
        epochs_trained=int(ep),
        train_loss=float(last_eval["loss"]),
        train_acc=float(last_eval["acc"]),
        stopped_early=bool(stopped_early),
        initial_lr=float(lr),
        initial_wd=float(weight_decay),
        continuing_lr=float(lr_cont),
        continuing_wd=float(wd_cont),
    )


def train_sam(
    *,
    model: nn.Module,
    train_ds: Dataset,
    eval_ds: Dataset,
    device: torch.device,
    epochs: int,
    lr: float,
    rho: float,
    adaptive: bool = False,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 128,
    num_workers: int = 2,
    stop_train_acc: float = 0.0,
    log_prefix: str = "[train-sam]",
    precision: str = "fp32",
    lr_schedule: str = "cosine",
) -> TrainResult:
    model = model.to(device)
    sam_opt = sam_train.SAM(
        model.parameters(),
        torch.optim.SGD,
        rho=float(rho),
        adaptive=bool(adaptive),
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    # SAM shares param_groups with its base optimizer; schedule that.
    scheduler = _build_lr_scheduler(sam_opt.base_optimizer, kind=lr_schedule, epochs=int(epochs))
    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )
    last_eval = {"loss": float("nan"), "acc": float("nan")}
    stopped_early = False
    ep = 0
    for ep in range(1, max(1, int(epochs)) + 1):
        model.train()
        with autocast_ctx(device, precision):
            for batch in train_loader:
                df_module.sam_update_on_batch(model, sam_opt, batch, device)
        ev = _train_eval_acc(model, eval_ds, device=device, batch_size=512,
                             num_workers=num_workers, precision=precision)
        last_eval = {"loss": ev["test_loss"], "acc": ev["test_acc"]}
        cur_lr = float(sam_opt.base_optimizer.param_groups[0].get("lr", lr))
        print(f"{log_prefix} ep={ep:03d} lr={cur_lr:.5g} train_loss={last_eval['loss']:.4f} train_acc={last_eval['acc']:.4f}", flush=True)
        if scheduler is not None:
            scheduler.step()
        if float(stop_train_acc) > 0.0 and float(last_eval["acc"]) >= float(stop_train_acc):
            stopped_early = True
            print(f"{log_prefix} early-stop: train_acc>={stop_train_acc:.4f}", flush=True)
            break

    base = sam_opt.base_optimizer
    lr_cont = float(base.param_groups[0].get("lr", lr))
    wd_cont = float(base.param_groups[0].get("weight_decay", weight_decay))
    return TrainResult(
        model=model,
        epochs_trained=int(ep),
        train_loss=float(last_eval["loss"]),
        train_acc=float(last_eval["acc"]),
        stopped_early=bool(stopped_early),
        initial_lr=float(lr),
        initial_wd=float(weight_decay),
        continuing_lr=float(lr_cont),
        continuing_wd=float(wd_cont),
    )


@dataclass
class ProbeArgs:
    probe_steps: int = 32
    probe_batch_size: int = 8
    step_train_eval_size: int = 64
    sam_step_batch_size: int = 8
    sam_rho: float = 0.05
    sam_lr: float = 1e-4
    sam_weight_decay: float = 0.1
    sam_momentum: float = 0.9
    sam_adaptive: bool = True
    sam_base_opt: str = "sgd"
    flat_r: float = 1.0
    flat_delta: float = 1.0
    flat_norm: str = "l2"
    record_every_steps: int = 0
    record_points: int = 10
    auto_microbatch_size: int = 64
    num_workers: int = 0
    show_progress: bool = False
    include_full_batch_step: bool = True
    include_unfreeze: bool = False
    min_steps: int = 2
    min_step_index: int = 0
    train_params_source: str = "specified"
    directional_flat_col: str = "flat_primal_freeze"


def probe_and_score(
    *,
    model: nn.Module,
    train_ds: Dataset,
    test_ds: Dataset,
    out_dir: Path,
    run_id: str,
    device: torch.device,
    meta: Dict[str, Any],
    training_lr: float,
    training_wd: float,
    continuing_lr: float,
    continuing_wd: float,
    probe: ProbeArgs = ProbeArgs(),
    precision: str = "fp32",
) -> Dict[str, Any]:
    """Run the directional sharpness probe and return the default DF scores."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _patch_set_device_for_cpu_if_needed(device)
    device_index = 0
    if device.type == "cuda":
        device_index = int(device.index or 0)

    probe_model = copy.deepcopy(model)
    sam_params = [p for p in probe_model.parameters() if p.requires_grad]
    flat_params = list(sam_params)

    cfg = df_module.ProbeConfig(
        out_dir=str(out_dir),
        probe_steps=int(probe.probe_steps),
        flat_norm=str(probe.flat_norm).strip().lower(),
        flat_r=float(probe.flat_r),
        flat_delta=float(probe.flat_delta),
        sam_rho=float(probe.sam_rho),
        sam_adaptive=bool(probe.sam_adaptive),
        sam_momentum=float(probe.sam_momentum),
        sam_lr=float(probe.sam_lr),
        sam_weight_decay=float(probe.sam_weight_decay),
        include_full_batch_step=bool(probe.include_full_batch_step),
        num_workers=int(probe.num_workers),
        device_index=int(device_index),
        include_unfreeze=bool(probe.include_unfreeze),
        show_progress=bool(probe.show_progress),
        probe_step_batch_size=int(probe.probe_batch_size),
        step_train_eval_size=int(probe.step_train_eval_size),
        sam_step_batch_size=int(probe.sam_step_batch_size),
        probe_epochs=0,
        probe_extra_steps=0,
        record_every_steps=int(probe.record_every_steps),
        record_points=int(probe.record_points),
        sam_base_opt=str(probe.sam_base_opt),
        num_gpus=1,
        max_concurrent_probes=1,
        auto_microbatch_size=int(probe.auto_microbatch_size),
    )

    print(f"[probe_and_score] cfg={asdict(cfg)}", flush=True)

    meta_full = dict(meta)
    meta_full.setdefault("probe_param_source", str(probe.train_params_source))
    meta_full.setdefault("lr", float(probe.sam_lr))
    meta_full.setdefault("wd", float(probe.sam_weight_decay))

    with autocast_ctx(device, precision):
        probe_csv = df_module.run_probe(
            model=probe_model,
            train_ds=train_ds,
            val_ds=test_ds,
            cfg=cfg,
            run_id=run_id,
            meta=meta_full,
            sam_params=sam_params,
            flat_params=flat_params,
            enable_grads_for_sam=True,
            enable_grads_for_flatness=True,
            train_params_source=str(probe.train_params_source),
            training_lr=float(training_lr),
            training_wd=float(training_wd),
            continuing_lr=float(continuing_lr),
            continuing_wd=float(continuing_wd),
            collate_fn=None,
        )

    score_cfg = df_module.DFConfig(
        probe_csv=str(probe_csv),
        target_columns=["final_test_acc"],
        out_dir=str(Path(out_dir) / "scores"),
        min_steps=max(2, int(probe.min_steps)),
        min_step_index=int(probe.min_step_index),
    )
    score_df, _ = df_module.compute_df(score_cfg)

    flat_col = str(probe.directional_flat_col).strip() or "flat_primal_freeze"
    if not score_df.empty and "flat_col" in score_df.columns:
        candidates = [flat_col, "flat_primal_freeze", "flat_primal"]
        chosen = None
        for c in candidates:
            if bool((score_df["flat_col"].astype(str) == c).any()):
                chosen = c
                break
        if chosen is None:
            chosen = str(score_df["flat_col"].iloc[0])
        scored = score_df[score_df["flat_col"].astype(str) == chosen].copy()
    else:
        chosen = flat_col
        scored = score_df

    out: Dict[str, Any] = {"probe_csv": str(probe_csv), "flat_col": str(chosen)}
    for col in DEFAULT_SCORE_COLUMNS:
        if not scored.empty and col in scored.columns:
            out[col] = float(scored[col].iloc[-1])
        else:
            out[col] = float("nan")
    return out


def summarize_pair(
    *,
    benign: Dict[str, Any],
    faulty: Dict[str, Any],
) -> pd.DataFrame:
    """Side-by-side benign/faulty metrics + diff, including all DF score columns."""
    rows = []
    preferred = ["test_acc", "test_loss", "train_acc", "train_loss",
                 "pre_quant_test_acc", "post_quant_test_acc",
                 "observed_test_acc", "full_test_acc"]
    seen: set = set()

    def _row(key: str) -> None:
        if key in seen:
            return
        seen.add(key)
        b = benign.get(key, float("nan"))
        f = faulty.get(key, float("nan"))
        try:
            diff = float(f) - float(b)
        except Exception:
            diff = float("nan")
        rows.append({
            "metric": key,
            "benign": b,
            "faulty": f,
            "diff_faulty_minus_benign": diff,
        })

    for key in preferred:
        if key in benign or key in faulty:
            _row(key)
    for key in DEFAULT_SCORE_COLUMNS:
        if key in benign or key in faulty:
            _row(key)
    extras = sorted((set(benign.keys()) | set(faulty.keys())) - seen - {"probe_csv", "flat_col"})
    for key in extras:
        b = benign.get(key, None)
        f = faulty.get(key, None)
        if isinstance(b, (int, float)) or isinstance(f, (int, float)):
            _row(key)
    return pd.DataFrame(rows)


# ----------------------------- Sweep support -----------------------------


def _csv_split(s: str) -> List[str]:
    return [tok.strip() for tok in str(s).split(",") if tok.strip()]


def parse_csv_ints(s: str) -> List[int]:
    return [int(tok) for tok in _csv_split(s)]


def parse_csv_floats(s: str) -> List[float]:
    return [float(tok) for tok in _csv_split(s)]


def parse_csv_strs(s: str) -> List[str]:
    return list(_csv_split(s))


def parse_csv_bools(s: str) -> List[bool]:
    out: List[bool] = []
    for tok in _csv_split(s):
        t = tok.lower()
        if t in {"true", "t", "1", "yes", "y", "on"}:
            out.append(True)
        elif t in {"false", "f", "0", "no", "n", "off"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean token: {tok!r}")
    return out


def parse_csv_flat_norms(s: str) -> List[str]:
    out = []
    for tok in _csv_split(s):
        t = tok.lower()
        if t not in {"l1", "l2"}:
            raise ValueError(f"--flat-norm must be l1 or l2, got {tok!r}")
        out.append(t)
    return out


def parse_csv_base_opts(s: str) -> List[str]:
    out = []
    for tok in _csv_split(s):
        t = tok.lower()
        if t not in {"sgd", "adamw"}:
            raise ValueError(f"--sam-base-opt must be sgd or adamw, got {tok!r}")
        out.append(t)
    return out


def parse_df_scores(s: str) -> List[str]:
    """Validate a CSV list of DF score variants against DEFAULT_SCORE_COLUMNS."""
    valid = set(DEFAULT_SCORE_COLUMNS)
    items = [tok.strip() for tok in str(s).split(",") if tok.strip()]
    if not items:
        return list(DEFAULT_SCORE_COLUMNS)
    bad = [x for x in items if x not in valid]
    if bad:
        raise ValueError(f"Unknown DF score variants: {bad}. Valid: {sorted(valid)}")
    return items


def add_probe_sweep_args(
    parser: argparse.ArgumentParser,
    *,
    sam_base_opt_default: str = "sgd",
) -> None:

    default_opt = str(sam_base_opt_default).strip().lower()
    if default_opt not in {"sgd", "adamw"}:
        raise ValueError(f"sam_base_opt_default must be sgd or adamw, got {sam_base_opt_default!r}")
    # --- swept (CSV) ---
    parser.add_argument("--probe-steps", type=parse_csv_ints, default=[16,32,64],
                        help="CSV-sweepable T (probe steps).")
    parser.add_argument("--probe-batch-size", type=parse_csv_ints, default=[4,8,16,32],
                        help="CSV-sweepable batch size for the per-step sharpness measurement.")
    parser.add_argument("--sam-step-batch-size", type=parse_csv_ints, default=[4],
                        help="CSV-sweepable batch size for the SAM update between measurements.")
    parser.add_argument("--step-train-eval-size", type=parse_csv_ints, default=[64],
                        help="CSV-sweepable batch size used to log train loss/acc at each recorded step.")
    parser.add_argument("--sam-rho", type=parse_csv_floats, default=[0.05,0.1,0.01],
                        help="CSV-sweepable SAM rho during probing.")
    parser.add_argument("--sam-lr", type=parse_csv_floats, default=[1e-3,1e-2,4e-2,4e-3,3e-3,7e-3],
                        help="CSV-sweepable SAM lr during probing.")
    parser.add_argument("--sam-adaptive", type=parse_csv_bools, default=[True,False],
                        help="CSV-sweepable adaptive-sharpness flag (e.g. 'true' or 'true,false').")
    parser.add_argument("--sam-base-opt", type=parse_csv_base_opts, default=[default_opt],
                        help="CSV-sweepable base optimizer wrapped by SAM during probing (sgd or adamw).")
    parser.add_argument("--flat-norm", type=parse_csv_flat_norms, default=["l2"],
                        help="CSV-sweepable flatness norm (l1 or l2).")
    parser.add_argument("--record-points", type=parse_csv_ints, default=[16],
                        help="CSV-sweepable number of recorded probe points along the trajectory.")
    # --- scalar (applied uniformly to every sweep) ---
    parser.add_argument("--sam-weight-decay", type=float, default=0.0,
                        help="Weight decay for the base optimizer wrapped by SAM during probing.")
    parser.add_argument("--sam-momentum", type=float, default=0.9,
                        help="Momentum for the SGD base optimizer during probing (ignored for adamw).")
    parser.add_argument("--flat-r", type=float, default=1.0,
                        help="Scale factor r in flat_primal = r * grad_norm.")
    parser.add_argument("--flat-delta", type=float, default=1.0,
                        help="Loss tolerance delta used by flat_dual.")
    parser.add_argument("--df-scores", type=str, default=",".join(DEFAULT_SCORE_COLUMNS),
                        help=("Comma-separated DF series scores to include in best-per-score and "
                              "all-sweeps tables. Defaults to all 9 variants "
                              "(FLAT_STD/VAR/RANGE, FLAT_OVER_LOSS_*, FLAT2_OVER_LOSS_*)."))


@dataclass(frozen=True)
class SweepConfig:
    # Swept knobs (CSV-typed in argparse).
    probe_steps: int
    probe_batch_size: int
    sam_step_batch_size: int
    step_train_eval_size: int
    sam_rho: float
    sam_lr: float
    sam_adaptive: bool
    sam_base_opt: str
    flat_norm: str
    record_points: int
    # Scalar knobs (uniform across all sweeps).
    sam_weight_decay: float = 0.1
    sam_momentum: float = 0.9
    flat_r: float = 1.0
    flat_delta: float = 1.0

    def to_probe_args(self) -> ProbeArgs:
        return ProbeArgs(
            probe_steps=int(self.probe_steps),
            probe_batch_size=int(self.probe_batch_size),
            sam_step_batch_size=int(self.sam_step_batch_size),
            step_train_eval_size=int(self.step_train_eval_size),
            sam_rho=float(self.sam_rho),
            sam_lr=float(self.sam_lr),
            sam_adaptive=bool(self.sam_adaptive),
            sam_base_opt=str(self.sam_base_opt),
            flat_norm=str(self.flat_norm),
            record_points=int(self.record_points),
            sam_weight_decay=float(self.sam_weight_decay),
            sam_momentum=float(self.sam_momentum),
            flat_r=float(self.flat_r),
            flat_delta=float(self.flat_delta),
        )

    def short_id(self) -> str:
        return (
            f"ps{self.probe_steps}_pb{self.probe_batch_size}_sb{self.sam_step_batch_size}"
            f"_se{self.step_train_eval_size}_rh{self.sam_rho:g}_lr{self.sam_lr:g}"
            f"_ad{int(self.sam_adaptive)}_opt{self.sam_base_opt}"
            f"_fn{self.flat_norm}_rp{self.record_points}"
        )


def build_sweeps(args: argparse.Namespace) -> List[SweepConfig]:
    swd = float(getattr(args, "sam_weight_decay", 0.1))
    smom = float(getattr(args, "sam_momentum", 0.9))
    fr = float(getattr(args, "flat_r", 1.0))
    fd = float(getattr(args, "flat_delta", 1.0))
    grids = (
        list(args.probe_steps), list(args.probe_batch_size),
        list(args.sam_step_batch_size), list(args.step_train_eval_size),
        list(args.sam_rho), list(args.sam_lr),
        list(args.sam_adaptive), list(args.sam_base_opt),
        list(args.flat_norm), list(args.record_points),
    )
    seen: set = set()
    out: List[SweepConfig] = []
    for combo in itertools.product(*grids):
        cfg = SweepConfig(
            probe_steps=combo[0],
            probe_batch_size=combo[1],
            sam_step_batch_size=combo[2],
            step_train_eval_size=combo[3],
            sam_rho=combo[4],
            sam_lr=combo[5],
            sam_adaptive=combo[6],
            sam_base_opt=combo[7],
            flat_norm=combo[8],
            record_points=combo[9],
            sam_weight_decay=swd,
            sam_momentum=smom,
            flat_r=fr,
            flat_delta=fd,
        )
        if cfg in seen:
            continue
        seen.add(cfg)
        out.append(cfg)
    return out


def best_per_score(
    sweep_results: List[Tuple[SweepConfig, Dict[str, Any], Dict[str, Any]]],
    *, score_columns: List[str] | None = None,
) -> pd.DataFrame:
    cols = list(score_columns) if score_columns is not None else list(DEFAULT_SCORE_COLUMNS)
    rows = []
    for col in cols:
        best_sweep = None
        best_ratio = float("inf")
        best_b = float("nan")
        best_f = float("nan")
        for sweep, b, f in sweep_results:
            bv = float(b.get(col, float("nan")))
            fv = float(f.get(col, float("nan")))
            if not (np.isfinite(bv) and np.isfinite(fv)):
                continue
            if fv == 0.0:
                continue
            ratio = bv / fv
            if ratio < best_ratio:
                best_ratio = ratio
                best_sweep = sweep
                best_b = bv
                best_f = fv
        row: Dict[str, Any] = {
            "score": col,
            "best_ratio": best_ratio if best_sweep is not None else float("nan"),
            "benign": best_b,
            "faulty": best_f,
        }
        if best_sweep is not None:
            row["sweep_id"] = best_sweep.short_id()
            row.update(asdict(best_sweep))
        else:
            row["sweep_id"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def all_sweeps_table(
    sweep_results: List[Tuple[SweepConfig, Dict[str, Any], Dict[str, Any]]],
    *, score_columns: List[str] | None = None,
) -> pd.DataFrame:
    """Long-form table with one row per (sweep, score variant)."""
    cols = list(score_columns) if score_columns is not None else list(DEFAULT_SCORE_COLUMNS)
    rows = []
    for sweep, b, f in sweep_results:
        for col in cols:
            bv = float(b.get(col, float("nan")))
            fv = float(f.get(col, float("nan")))
            ratio = (bv / fv) if (np.isfinite(bv) and np.isfinite(fv) and fv != 0.0) else float("nan")
            rows.append({
                "sweep_id": sweep.short_id(),
                "score": col,
                "benign": bv,
                "faulty": fv,
                "ratio_benign_over_faulty": ratio,
                **asdict(sweep),
            })
    return pd.DataFrame(rows)


# ------------------------- Static sharpness -------------------------


def static_sharpness_once(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
    adaptive: bool = True,
    norm: str = "l2",
) -> float:
    """Full-dataset ASAM sharpness on `dataset`.
    Returns the L1 or L2 norm of the full-batch gradient (optionally weighted
    by |w| for ASAM).
    """
    from probe_common import fullbatch_sharpness_once  # noqa: E402

    stats = fullbatch_sharpness_once(
        model, dataset,
        device=device,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        adaptive=bool(adaptive),
    )
    norm_l = str(norm).strip().lower()
    if norm_l == "l1":
        return float(stats["fullbatch_sharp_l1"])
    if norm_l == "l2":
        return float(stats["fullbatch_sharp_l2"])
    raise ValueError(f"Unsupported norm {norm!r} for static_sharpness_once; expected l1 or l2.")


def static_comparison_table(
    *,
    observed_acc_benign: float,
    observed_acc_faulty: float,
    static_benign: float,
    static_faulty: float,
    benign_label: str = "benign",
    faulty_label: str = "faulty",
) -> pd.DataFrame:
    def _ratio(b: float, f: float) -> float:
        try:
            bv, fv = float(b), float(f)
        except Exception:
            return float("nan")
        if not (np.isfinite(bv) and np.isfinite(fv)) or fv == 0.0:
            return float("nan")
        return bv / fv

    rows = [
        {
            "metric": "observed_acc",
            benign_label: float(observed_acc_benign),
            faulty_label: float(observed_acc_faulty),
            "ratio_benign_over_faulty": _ratio(observed_acc_benign, observed_acc_faulty),
        },
        {
            "metric": "static_sharpness",
            benign_label: float(static_benign),
            faulty_label: float(static_faulty),
            "ratio_benign_over_faulty": _ratio(static_benign, static_faulty),
        },
    ]
    return pd.DataFrame(rows)
