"""probing / d-flatness scoring utilities."""
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

# ----------------------------- Parsing helpers -----------------------------


def _parse_csv_list(raw: str, cast, *, item_name: str) -> List[Any]:
    items = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not items:
        raise ValueError(f"Expected a comma-separated list of {item_name} values.")
    out: List[Any] = []
    for item in items:
        try:
            out.append(cast(item))
        except Exception as exc:
            raise ValueError(f"Invalid {item_name} value '{item}' in '{raw}'.") from exc
    return out


def _parse_csv_ints(raw: str) -> List[int]:
    return _parse_csv_list(raw, int, item_name="integer")


def _parse_csv_floats(raw: str) -> List[float]:
    return _parse_csv_list(raw, float, item_name="float")


def _parse_csv_strings(raw: str) -> List[str]:
    return _parse_csv_list(raw, str, item_name="string")


def _parse_bool_token(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean token: '{text}'")


def _parse_csv_bools(raw: str) -> List[bool]:
    return _parse_csv_list(raw, _parse_bool_token, item_name="boolean")


def _to_float_or_nan(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _fmt_float_tag(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


# ----------------------------- Dataset/model helpers -----------------------------


class XYDataset(Dataset):
    """Adapt datasets that return (x, y, idx) into (x, y)."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return int(len(self.base))

    def __getitem__(self, index: int):
        item = self.base[index]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return item[0], item[1]
        raise TypeError(f"Unsupported dataset item format: {type(item)}")


def _split_xy(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        if "labels" in batch:
            return batch["input_ids"], batch["labels"]
        if "label" in batch:
            return batch["input_ids"], batch["label"]
    raise TypeError(f"Unsupported batch type for (x, y) split: {type(batch)}")


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _clone_model_cpu(model: nn.Module) -> nn.Module:
    with torch.no_grad():
        clone = copy.deepcopy(_unwrap_model(model)).cpu()
    return clone


def _extract_optimizer_lr_wd(
    optimizer: Optional[torch.optim.Optimizer],
    *,
    fallback_lr: float,
    fallback_wd: float,
) -> Tuple[float, float]:
    if optimizer is None:
        return float(fallback_lr), float(fallback_wd)
    groups = list(getattr(optimizer, "param_groups", []))
    if not groups:
        return float(fallback_lr), float(fallback_wd)
    lr = float(groups[0].get("lr", fallback_lr))
    wd = float(groups[0].get("weight_decay", fallback_wd))
    if not np.isfinite(lr) or lr <= 0:
        lr = float(fallback_lr)
    if not np.isfinite(wd) or wd < 0:
        wd = float(fallback_wd)
    return float(lr), float(wd)


@torch.no_grad()
def evaluate_loss_acc(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(int(num_workers) > 0),
    )
    ce_sum = nn.CrossEntropyLoss(reduction="sum")

    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_n = 0

    for batch in loader:
        xb, yb = _split_xy(batch)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).long()

        logits = model(xb)
        total_loss += float(ce_sum(logits, yb).item())
        total_correct += float((logits.argmax(dim=1) == yb).float().sum().item())
        total_n += int(yb.numel())

    denom = max(1, total_n)
    out = {
        "test_loss": float(total_loss / denom),
        "test_acc": float(total_correct / denom),
        "test_n": float(total_n),
    }
    model.to(torch.device("cpu"))
    return out


def fullbatch_sharpness_once(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    adaptive: bool,
) -> Dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(int(num_workers) > 0),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        params = [p for _, p in model.named_parameters()]

    model = model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)

    total_n = int(len(dataset))
    if total_n <= 0:
        raise ValueError("Cannot compute full-batch sharpness on an empty dataset.")

    total_loss = 0.0
    total_correct = 0.0

    for batch in loader:
        xb, yb = _split_xy(batch)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).long()

        logits = model(xb)
        loss_mean = nn.functional.cross_entropy(logits, yb, reduction="mean")
        bs = int(yb.shape[0])

        total_loss += float(loss_mean.detach().item()) * bs
        total_correct += float((logits.argmax(dim=1) == yb).float().sum().item())

        # Gradient of full-batch mean loss.
        (loss_mean * (float(bs) / float(total_n))).backward()

    sum_l1 = 0.0
    sum_l2_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        term = p.detach() * g if adaptive else g
        sum_l1 += float(term.abs().sum().item())
        sum_l2_sq += float(term.square().sum().item())

    grad_l1 = float(sum_l1)
    grad_l2 = float(math.sqrt(max(0.0, sum_l2_sq)))

    out = {
        "fullbatch_loss": float(total_loss / float(total_n)),
        "fullbatch_acc": float(total_correct / float(total_n)),
        "fullbatch_sharp_l1": grad_l1,
        "fullbatch_sharp_l2": grad_l2,
    }
    model.to(torch.device("cpu"))
    return out


# ----------------------------- Trend metrics -----------------------------


@dataclass(frozen=True)
class TrendConfig:
    min_step: int
    min_points: int
    window_sizes: Tuple[int, ...]
    window_trim_fractions: Tuple[float, ...]
    slope_smoother_params: Tuple[float, ...]
    cumulative_thresholds: Tuple[float, ...]
    sign_thresholds: Tuple[float, ...]
    monotone_smoother_params: Tuple[float, ...]


TREND_WINDOW_AGGREGATORS: Tuple[str, ...] = ("median", "trimmed_mean")
TREND_SLOPE_SMOOTHERS: Tuple[str, ...] = ("none", "moving_average", "moving_median", "ema")
TREND_MONOTONE_SMOOTHERS: Tuple[str, ...] = ("none", "moving_average", "moving_median", "ema")


def _trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    tf = max(0.0, min(0.49, float(trim_fraction)))
    if tf <= 0.0:
        return float(np.mean(arr))
    n = int(arr.size)
    k = int(math.floor(tf * n))
    if 2 * k >= n:
        return float(np.mean(arr))
    arr = np.sort(arr)[k : n - k]
    return float(np.mean(arr))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    if w <= 1 or len(values) <= 2:
        return values.copy()
    if w >= len(values):
        mean_val = float(np.mean(values))
        return np.full_like(values, mean_val, dtype=float)
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(values, kernel, mode="same")


def _moving_median(values: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    if w <= 1 or len(values) <= 2:
        return values.copy()
    half = w // 2
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def _ema(values: np.ndarray, param: float) -> np.ndarray:
    if len(values) <= 1:
        return values.copy()
    if param <= 0:
        return values.copy()
    if param > 1.0:
        alpha = 2.0 / (float(param) + 1.0)
    else:
        alpha = float(param)
    alpha = max(1e-6, min(1.0, alpha))
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _smooth(values: np.ndarray, *, mode: str, param: float) -> np.ndarray:
    key = str(mode).strip().lower()
    if key == "none":
        return values.copy()
    if key == "moving_average":
        return _moving_average(values, int(round(param)))
    if key == "moving_median":
        return _moving_median(values, int(round(param)))
    if key == "ema":
        return _ema(values, float(param))
    raise ValueError(
        f"Unsupported trend smoother '{mode}'. Use one of: none, moving_average, moving_median, ema."
    )


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Keep x/y aligned even if a smoother emitted an unexpected length.
    n_common = int(min(x.size, y.size))
    x = x[:n_common]
    y = y[:n_common]
    n = int(x.size)
    if n < 2:
        return float("nan")
    slopes: List[np.ndarray] = []
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        valid = dx != 0
        if not np.any(valid):
            continue
        dy = y[i + 1 :] - y[i]
        slopes.append(dy[valid] / dx[valid])
    if not slopes:
        return float("nan")
    all_slopes = np.concatenate(slopes)
    all_slopes = all_slopes[np.isfinite(all_slopes)]
    if all_slopes.size == 0:
        return float("nan")
    return float(np.median(all_slopes))


def _window_stat(values: np.ndarray, *, agg: str, trim_fraction: float) -> float:
    key = str(agg).strip().lower()
    if key == "median":
        return float(np.median(values))
    if key == "trimmed_mean":
        return _trimmed_mean(values, trim_fraction)
    raise ValueError(f"Unsupported trend window aggregator '{agg}'.")


def _sanitize_token(text: object) -> str:
    s = str(text)
    out_chars: List[str] = []
    for ch in s:
        out_chars.append(ch if ch.isalnum() else "_")
    out = "".join(out_chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "item"


def _to_odd_int(value: int) -> int:
    iv = int(value)
    return iv if (iv % 2 == 1) else (iv + 1)


def _scaled_trend_values_from_probe_steps(probe_steps: int) -> Dict[str, List[float]]:
    n = max(2, int(probe_steps))

    window_candidates = [
        max(2, int(round(0.2 * n))),
        max(2, int(round(0.6 * n))),
    ]
    window_cap = max(2, n)
    window_sizes = sorted(set(max(2, min(int(v), window_cap)) for v in window_candidates))
    if len(window_sizes) == 1 and window_cap >= 3:
        alt = max(2, min(window_cap, window_sizes[0] + 1))
        if alt != window_sizes[0]:
            window_sizes.append(alt)
            window_sizes = sorted(set(window_sizes))

    slope_candidates = [
        _to_odd_int(max(3, int(round(0.3 * n)))),
        _to_odd_int(max(3, int(round(0.5 * n)))),
    ]
    slope_cap = max(3, n if n % 2 == 1 else n - 1)
    slope_params = sorted(set(max(3, min(int(v), slope_cap)) for v in slope_candidates))
    if len(slope_params) == 1 and slope_cap >= 5:
        alt = max(3, min(slope_cap, slope_params[0] + 2))
        if alt != slope_params[0]:
            slope_params.append(alt)
            slope_params = sorted(set(slope_params))

    inv_n = 1.0 / float(max(1, n))
    thresholds = sorted(set([0.0, 0.01 * inv_n, 0.05 * inv_n]))

    return {
        "window_sizes": [int(v) for v in window_sizes],
        "slope_params": [float(v) for v in slope_params],
        "cumulative_thresholds": [float(v) for v in thresholds],
        "sign_thresholds": [float(v) for v in thresholds],
        "monotone_params": [float(v) for v in slope_params],
    }


def _score_windowed_drop(values: np.ndarray, *, window_size: int, aggregator: str, trim_fraction: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")

    w = max(1, min(int(window_size), len(x)))
    start_stat = _window_stat(x[:w], agg=aggregator, trim_fraction=trim_fraction)
    end_stat = _window_stat(x[-w:], agg=aggregator, trim_fraction=trim_fraction)
    if not np.isfinite(start_stat) or not np.isfinite(end_stat):
        return float("nan")
    return float(start_stat - end_stat)


def _score_thresholded_cumulative(values: np.ndarray, *, threshold: float, mode: str) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")

    thr = max(0.0, float(threshold))
    diff = np.diff(x)
    dec = np.maximum(-diff - thr, 0.0)
    inc = np.maximum(diff - thr, 0.0)
    dec_sum = float(np.sum(dec))
    inc_sum = float(np.sum(inc))

    if mode == "net":
        return float(dec_sum - inc_sum)
    if mode == "ratio":
        if dec_sum <= 0.0 and inc_sum <= 0.0:
            return 1.0
        return float(dec_sum / max(inc_sum, 1e-12))
    raise ValueError(f"Unsupported cumulative mode '{mode}'.")


def _score_sign_decrease_frequency(values: np.ndarray, *, threshold: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")

    thr = max(0.0, float(threshold))
    diff = np.diff(x)
    dec = int(np.sum(diff < -thr))
    inc = int(np.sum(diff > thr))
    meaningful = dec + inc
    if meaningful == 0:
        return 0.5
    return float(dec) / float(meaningful)


def _isotonic_increasing_fit(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()

    levels: List[float] = []
    weights: List[float] = []
    starts: List[int] = []
    ends: List[int] = []

    for i in range(n):
        levels.append(float(y[i]))
        weights.append(1.0)
        starts.append(i)
        ends.append(i)

        while len(levels) >= 2 and levels[-2] > levels[-1]:
            w1 = weights[-2]
            w2 = weights[-1]
            merged_w = w1 + w2
            merged_level = (levels[-2] * w1 + levels[-1] * w2) / merged_w
            levels[-2] = float(merged_level)
            weights[-2] = float(merged_w)
            ends[-2] = ends[-1]

            levels.pop()
            weights.pop()
            starts.pop()
            ends.pop()

    out = np.empty(n, dtype=float)
    for level, start, end in zip(levels, starts, ends):
        out[start : end + 1] = float(level)
    return out


def _isotonic_decreasing_fit(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    return -_isotonic_increasing_fit(-y)


def _score_monotone_denoised_drop(values: np.ndarray, *, smoother: str, smoother_param: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    smooth = _smooth(x, mode=smoother, param=smoother_param)
    fit = _isotonic_decreasing_fit(smooth)
    if len(fit) < 2 or not np.isfinite(fit[0]) or not np.isfinite(fit[-1]):
        return float("nan")
    return float(fit[0] - fit[-1])


def _build_trend_config(args: argparse.Namespace, *, probe_steps: int) -> TrendConfig:
    auto_scale = str(args.trend_auto_scale_from_steps).strip().lower() == "true"
    scaled = _scaled_trend_values_from_probe_steps(int(probe_steps)) if auto_scale and int(probe_steps) > 0 else None

    if args.trend_window_sizes is None:
        window_sizes = list(scaled["window_sizes"]) if scaled is not None else [5]
    else:
        window_sizes = _parse_csv_ints(str(args.trend_window_sizes))

    if args.trend_window_trim_fractions is None:
        window_trim_fractions = [0.1]
    else:
        window_trim_fractions = _parse_csv_floats(str(args.trend_window_trim_fractions))

    if args.trend_slope_smoother_params is None:
        slope_smoother_params = list(scaled["slope_params"]) if scaled is not None else [5.0]
    else:
        slope_smoother_params = _parse_csv_floats(str(args.trend_slope_smoother_params))

    if args.trend_cumulative_thresholds is None:
        cumulative_thresholds = list(scaled["cumulative_thresholds"]) if scaled is not None else [0.0]
    else:
        cumulative_thresholds = _parse_csv_floats(str(args.trend_cumulative_thresholds))

    if args.trend_sign_thresholds is None:
        sign_thresholds = list(scaled["sign_thresholds"]) if scaled is not None else [0.0]
    else:
        sign_thresholds = _parse_csv_floats(str(args.trend_sign_thresholds))

    if args.trend_monotone_smoother_params is None:
        monotone_smoother_params = list(scaled["monotone_params"]) if scaled is not None else [5.0]
    else:
        monotone_smoother_params = _parse_csv_floats(str(args.trend_monotone_smoother_params))

    if any(int(v) <= 0 for v in window_sizes):
        raise ValueError(f"--trend-window-sizes values must be > 0. Got: {window_sizes}")
    if any(float(v) < 0.0 or float(v) >= 0.5 for v in window_trim_fractions):
        raise ValueError(
            f"--trend-window-trim-fractions values must be in [0, 0.5). Got: {window_trim_fractions}"
        )
    if any(float(v) < 0.0 for v in cumulative_thresholds):
        raise ValueError(
            f"--trend-cumulative-thresholds values must be >= 0. Got: {cumulative_thresholds}"
        )
    if any(float(v) < 0.0 for v in sign_thresholds):
        raise ValueError(f"--trend-sign-thresholds values must be >= 0. Got: {sign_thresholds}")

    return TrendConfig(
        min_step=int(args.trend_min_step),
        min_points=int(args.trend_min_points),
        window_sizes=tuple(int(v) for v in window_sizes),
        window_trim_fractions=tuple(float(v) for v in window_trim_fractions),
        slope_smoother_params=tuple(float(v) for v in slope_smoother_params),
        cumulative_thresholds=tuple(float(v) for v in cumulative_thresholds),
        sign_thresholds=tuple(float(v) for v in sign_thresholds),
        monotone_smoother_params=tuple(float(v) for v in monotone_smoother_params),
    )


def compute_trend_scores(steps: np.ndarray, values: np.ndarray, cfg: TrendConfig) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(values) < max(2, int(cfg.min_points)):
        return out

    for aggregator in TREND_WINDOW_AGGREGATORS:
        for window_size in cfg.window_sizes:
            if str(aggregator).strip().lower() == "trimmed_mean":
                for trim_fraction in cfg.window_trim_fractions:
                    score = _score_windowed_drop(
                        values,
                        window_size=int(window_size),
                        aggregator=str(aggregator),
                        trim_fraction=float(trim_fraction),
                    )
                    col = (
                        f"TREND_WIN_DROP_{_sanitize_token(aggregator).upper()}_W{int(window_size)}"
                        f"_TR{_fmt_float_tag(float(trim_fraction))}"
                    )
                    out[col] = float(score)
            else:
                score = _score_windowed_drop(
                    values,
                    window_size=int(window_size),
                    aggregator=str(aggregator),
                    trim_fraction=float(cfg.window_trim_fractions[0]),
                )
                col = f"TREND_WIN_DROP_{_sanitize_token(aggregator).upper()}_W{int(window_size)}"
                out[col] = float(score)

    for smoother in TREND_SLOPE_SMOOTHERS:
        if str(smoother).strip().lower() == "none":
            smooth_vals = _smooth(values, mode=str(smoother), param=0.0)
            slope = _theil_sen_slope(steps, smooth_vals)
            col = f"TREND_SLOPE_THEIL_SEN_SM_{_sanitize_token(smoother).upper()}"
            out[col] = float(-slope) if np.isfinite(slope) else float("nan")
        else:
            for smoother_param in cfg.slope_smoother_params:
                smooth_vals = _smooth(values, mode=str(smoother), param=float(smoother_param))
                slope = _theil_sen_slope(steps, smooth_vals)
                col = (
                    f"TREND_SLOPE_THEIL_SEN_SM_{_sanitize_token(smoother).upper()}"
                    f"_P{_fmt_float_tag(float(smoother_param))}"
                )
                out[col] = float(-slope) if np.isfinite(slope) else float("nan")

    for threshold in cfg.cumulative_thresholds:
        thr_tag = _fmt_float_tag(float(threshold))
        out[f"TREND_CUM_NET_T{thr_tag}"] = float(
            _score_thresholded_cumulative(values, threshold=float(threshold), mode="net")
        )
        out[f"TREND_CUM_RATIO_T{thr_tag}"] = float(
            _score_thresholded_cumulative(values, threshold=float(threshold), mode="ratio")
        )

    for threshold in cfg.sign_thresholds:
        thr_tag = _fmt_float_tag(float(threshold))
        out[f"TREND_SIGN_DECR_FREQ_T{thr_tag}"] = float(
            _score_sign_decrease_frequency(values, threshold=float(threshold))
        )

    for smoother in TREND_MONOTONE_SMOOTHERS:
        if str(smoother).strip().lower() == "none":
            col = f"TREND_MONO_DROP_SM_{_sanitize_token(smoother).upper()}"
            out[col] = float(
                _score_monotone_denoised_drop(values, smoother=str(smoother), smoother_param=0.0)
            )
        else:
            for smoother_param in cfg.monotone_smoother_params:
                col = (
                    f"TREND_MONO_DROP_SM_{_sanitize_token(smoother).upper()}"
                    f"_P{_fmt_float_tag(float(smoother_param))}"
                )
                out[col] = float(
                    _score_monotone_denoised_drop(
                        values,
                        smoother=str(smoother),
                        smoother_param=float(smoother_param),
                    )
                )

    return out


def compute_directional_trend_rows(
    probe_df: pd.DataFrame,
    *,
    flat_col: str,
    trend_cfg: TrendConfig,
) -> pd.DataFrame:
    if probe_df.empty:
        return pd.DataFrame(columns=["flat_norm"])
    if "flat_norm" not in probe_df.columns:
        probe_df = probe_df.copy()
        probe_df["flat_norm"] = "l2"

    rows: List[Dict[str, Any]] = []

    for flat_norm, g0 in probe_df.groupby("flat_norm", sort=False):
        g = g0.copy()
        g["step"] = pd.to_numeric(g.get("step"), errors="coerce")
        g[flat_col] = pd.to_numeric(g.get(flat_col), errors="coerce")
        g["train_loss"] = pd.to_numeric(g.get("train_loss"), errors="coerce")

        g = g[g["step"].notna()]
        g = g[g["step"] >= int(trend_cfg.min_step)]
        g = g.sort_values("step")
        if g.empty:
            continue

        steps = g["step"].to_numpy(dtype=float)
        flat = g[flat_col].to_numpy(dtype=float)
        loss = g["train_loss"].to_numpy(dtype=float)

        valid = np.isfinite(steps) & np.isfinite(flat) & np.isfinite(loss)
        steps = steps[valid]
        flat = flat[valid]
        loss = loss[valid]
        if len(flat) < max(2, int(trend_cfg.min_points)):
            continue

        loss_eps = np.maximum(loss, 1e-12)
        variants: Dict[str, np.ndarray] = {
            "FLAT": flat,
            "FLAT_OVER_LOSS": flat / loss_eps,
            "FLAT2_OVER_LOSS": (flat * flat) / loss_eps,
        }

        row: Dict[str, Any] = {"flat_norm": str(flat_norm)}
        for variant_name, values in variants.items():
            scores = compute_trend_scores(steps=steps, values=values, cfg=trend_cfg)
            for key, value in scores.items():
                row[f"{key}__VAR_{variant_name}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------- Probe config -----------------------------


@dataclass(frozen=True)
class ProbeSweep:
    probe_steps: int
    probe_epochs: int
    probe_extra_steps: int
    probe_step_batch_size: int
    step_train_eval_size: int
    sam_step_batch_size: int
    sam_rho: float
    sam_lr: float
    sam_weight_decay: float
    sam_momentum: float
    sam_base_opt: str
    sam_adaptive: bool
    flat_r: float
    flat_delta: float
    record_every_steps: int
    record_points: int
    auto_microbatch_size: int

    def sweep_id(self) -> str:
        return (
            f"ps{self.probe_steps}_pe{self.probe_epochs}_px{self.probe_extra_steps}"
            f"_pbs{self.probe_step_batch_size}_seb{self.step_train_eval_size}_sam{self.sam_step_batch_size}"
            f"_rho{_fmt_float_tag(self.sam_rho)}_lr{_fmt_float_tag(self.sam_lr)}"
            f"_wd{_fmt_float_tag(self.sam_weight_decay)}_mom{_fmt_float_tag(self.sam_momentum)}"
            f"_opt{self.sam_base_opt}_ada{int(self.sam_adaptive)}"
            f"_fr{_fmt_float_tag(self.flat_r)}_fd{_fmt_float_tag(self.flat_delta)}"
            f"_re{self.record_every_steps}_rp{self.record_points}_amb{self.auto_microbatch_size}"
        )


def build_probe_sweeps(args: argparse.Namespace) -> List[ProbeSweep]:
    probe_steps = [int(v) for v in args.sweep_probe_steps]
    probe_epochs = [int(v) for v in args.sweep_probe_epochs]
    probe_extra_steps = [int(v) for v in args.sweep_probe_extra_steps]
    probe_step_bs = [int(v) for v in args.sweep_probe_step_batch_size]
    step_eval_bs = [int(v) for v in args.sweep_step_train_eval_size]
    sam_step_bs = [int(v) for v in args.sweep_sam_step_batch_size]
    sam_rho = [float(v) for v in args.sweep_sam_rho]
    sam_lr = [float(v) for v in args.sweep_sam_lr]
    sam_wd = [float(args.sam_weight_decay)]
    sam_mom = [float(args.sam_momentum)]
    sam_opt = [str(v).strip().lower() for v in args.sweep_sam_base_opt]
    sam_adaptive = [bool(args.sam_adaptive)]
    flat_r = [float(args.flat_r)]
    flat_delta = [float(args.flat_delta)]
    record_every = [int(v) for v in args.sweep_record_every_steps]
    record_points = [int(v) for v in args.sweep_record_points]
    auto_micro = [int(args.auto_microbatch_size)]

    bad_probe_steps = [v for v in probe_steps if int(v) < 0]
    bad_probe_epochs = [v for v in probe_epochs if int(v) < 0]
    bad_probe_extra_steps = [v for v in probe_extra_steps if int(v) < 0]
    bad_record_every = [v for v in record_every if int(v) < 0]
    bad_record_points = [v for v in record_points if int(v) < 0]
    bad_probe_step_bs = [v for v in probe_step_bs if int(v) <= 0]
    bad_step_eval_bs = [v for v in step_eval_bs if int(v) <= 0]
    bad_sam_step_bs = [v for v in sam_step_bs if int(v) <= 0]
    bad_sam_rho = [v for v in sam_rho if float(v) < 0.0]
    bad_sam_lr = [v for v in sam_lr if float(v) <= 0.0]
    bad_auto_micro = [v for v in auto_micro if int(v) <= 0]
    if bad_probe_steps:
        raise ValueError(f"Invalid --sweep-probe-steps value(s) (must be >= 0): {bad_probe_steps}")
    if bad_probe_epochs:
        raise ValueError(f"Invalid --sweep-probe-epochs value(s) (must be >= 0): {bad_probe_epochs}")
    if bad_probe_extra_steps:
        raise ValueError(
            f"Invalid --sweep-probe-extra-steps value(s) (must be >= 0): {bad_probe_extra_steps}"
        )
    if bad_record_every:
        raise ValueError(
            f"Invalid --sweep-record-every-steps value(s) (must be >= 0): {bad_record_every}"
        )
    if bad_record_points:
        raise ValueError(f"Invalid --sweep-record-points value(s) (must be >= 0): {bad_record_points}")
    if bad_probe_step_bs:
        raise ValueError(
            f"Invalid --sweep-probe-step-batch-size value(s) (must be > 0): {bad_probe_step_bs}"
        )
    if bad_step_eval_bs:
        raise ValueError(
            f"Invalid --sweep-step-train-eval-size value(s) (must be > 0): {bad_step_eval_bs}"
        )
    if bad_sam_step_bs:
        raise ValueError(
            f"Invalid --sweep-sam-step-batch-size value(s) (must be > 0): {bad_sam_step_bs}"
        )
    if bad_sam_rho:
        raise ValueError(f"Invalid --sweep-sam-rho value(s) (must be >= 0): {bad_sam_rho}")
    if bad_sam_lr:
        raise ValueError(f"Invalid --sweep-sam-lr value(s) (must be > 0): {bad_sam_lr}")
    if bad_auto_micro:
        raise ValueError(f"Invalid --auto-microbatch-size value(s) (must be > 0): {bad_auto_micro}")

    sweeps: List[ProbeSweep] = []
    for combo in itertools.product(
        probe_steps,
        probe_epochs,
        probe_extra_steps,
        probe_step_bs,
        step_eval_bs,
        sam_step_bs,
        sam_rho,
        sam_lr,
        sam_wd,
        sam_mom,
        sam_opt,
        sam_adaptive,
        flat_r,
        flat_delta,
        record_every,
        record_points,
        auto_micro,
    ):
        sweep = ProbeSweep(
            probe_steps=int(combo[0]),
            probe_epochs=int(combo[1]),
            probe_extra_steps=int(combo[2]),
            probe_step_batch_size=int(combo[3]),
            step_train_eval_size=int(combo[4]),
            sam_step_batch_size=int(combo[5]),
            sam_rho=float(combo[6]),
            sam_lr=float(combo[7]),
            sam_weight_decay=float(combo[8]),
            sam_momentum=float(combo[9]),
            sam_base_opt=str(combo[10]).strip().lower(),
            sam_adaptive=bool(combo[11]),
            flat_r=float(combo[12]),
            flat_delta=float(combo[13]),
            record_every_steps=int(combo[14]),
            record_points=int(combo[15]),
            auto_microbatch_size=int(combo[16]),
        )

        if sweep.sam_base_opt not in {"sgd", "adamw"}:
            raise ValueError(f"Unsupported sam_base_opt '{sweep.sam_base_opt}'. Use sgd/adamw.")
        if sweep.record_every_steps > 0 and sweep.record_points > 0:
            raise ValueError(
                "record_every_steps and record_points cannot both be > 0 in the same probe sweep."
            )
        sweeps.append(sweep)

    # De-duplicate exact copies while preserving order.
    unique: List[ProbeSweep] = []
    seen = set()
    for sweep in sweeps:
        key = asdict(sweep)
        frozen_key = tuple(sorted(key.items()))
        if frozen_key in seen:
            continue
        seen.add(frozen_key)
        unique.append(sweep)
    return unique




def load_df_module(dflat_root: Path):
    df_root = (dflat_root / "df_utils").resolve()
    if str(df_root) not in sys.path:
        sys.path.insert(0, str(df_root))
    import df as df_module  # noqa: E402

    return df_module


# ----------------------------- Probe run for one model -----------------------------


def run_probe_and_collect_scores(
    *,
    df_module,
    base_model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    sweep: ProbeSweep,
    run_id: str,
    out_dir: Path,
    args: argparse.Namespace,
    meta: Dict[str, Any],
    training_lr: float,
    training_wd: float,
    continuing_lr: float,
    continuing_wd: float,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    trend_cfg = _build_trend_config(args, probe_steps=int(sweep.probe_steps))

    source = str(args.train_params_source).strip().lower()
    if source not in {"specified", "initial", "continuing"}:
        raise ValueError(
            f"Unsupported train_params_source '{args.train_params_source}'. "
            "Use one of: specified, initial, continuing."
        )
    if source == "specified":
        chosen_lr = float(sweep.sam_lr)
        chosen_wd = float(sweep.sam_weight_decay)
    elif source == "initial":
        chosen_lr = float(training_lr)
        chosen_wd = float(training_wd)
    else:
        chosen_lr = float(continuing_lr)
        chosen_wd = float(continuing_wd)

    meta_with_probe_params = dict(meta)
    meta_with_probe_params["probe_param_source"] = source
    meta_with_probe_params["lr"] = chosen_lr
    meta_with_probe_params["wd"] = chosen_wd

    probe_cfg = df_module.ProbeConfig(
        out_dir=str(out_dir),
        probe_steps=int(sweep.probe_steps),
        flat_norm=str(args.flat_norm).strip().lower(),
        flat_r=float(sweep.flat_r),
        flat_delta=float(sweep.flat_delta),
        sam_rho=float(sweep.sam_rho),
        sam_adaptive=bool(sweep.sam_adaptive),
        sam_momentum=float(sweep.sam_momentum),
        sam_lr=float(sweep.sam_lr),
        sam_weight_decay=float(sweep.sam_weight_decay),
        include_full_batch_step=bool(args.include_full_batch_step),
        num_workers=int(args.num_workers),
        device_index=int(args.probe_device_index_effective),
        include_unfreeze=bool(args.include_unfreeze),
        show_progress=bool(args.show_progress),
        probe_step_batch_size=int(sweep.probe_step_batch_size),
        step_train_eval_size=int(sweep.step_train_eval_size),
        sam_step_batch_size=int(sweep.sam_step_batch_size),
        probe_epochs=int(sweep.probe_epochs),
        probe_extra_steps=int(sweep.probe_extra_steps),
        record_every_steps=int(sweep.record_every_steps),
        record_points=int(sweep.record_points),
        sam_base_opt=str(sweep.sam_base_opt),
        num_gpus=1,
        max_concurrent_probes=1,
        auto_microbatch_size=int(sweep.auto_microbatch_size),
    )

    probe_model = copy.deepcopy(base_model)
    sam_params = [p for p in probe_model.parameters() if p.requires_grad]
    flat_params = [p for p in probe_model.parameters() if p.requires_grad]

    probe_csv = df_module.run_probe(
        model=probe_model,
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=probe_cfg,
        run_id=run_id,
        meta=meta_with_probe_params,
        sam_params=sam_params,
        flat_params=flat_params,
        enable_grads_for_sam=True,
        enable_grads_for_flatness=True,
        train_params_source=source,
        training_lr=float(training_lr),
        training_wd=float(training_wd),
        continuing_lr=float(continuing_lr),
        continuing_wd=float(continuing_wd),
        collate_fn=None,
    )

    probe_path = Path(probe_csv)
    probe_df = pd.read_csv(probe_path)

    score_cfg = df_module.DFConfig(
        probe_csv=str(probe_path),
        target_columns=["final_test_acc"],
        out_dir=str(out_dir / "scores"),
        min_steps=max(2, int(args.min_steps)),
        min_step_index=int(args.min_step_index),
    )
    try:
        score_df, _ = df_module.compute_df(score_cfg)
    except KeyError as exc:
        # Backward-compatibility with older df.py variants that still expose test_acc.
        if "final_test_acc" not in str(exc):
            raise
        score_cfg = df_module.DFConfig(
            probe_csv=str(probe_path),
            target_columns=["test_acc"],
            out_dir=str(out_dir / "scores"),
            min_steps=max(2, int(args.min_steps)),
            min_step_index=int(args.min_step_index),
        )
        score_df, _ = df_module.compute_df(score_cfg)

    flat_col = str(args.directional_flat_col).strip()
    if not flat_col:
        flat_col = "flat_primal_freeze"

    candidate_cols = [flat_col, "flat_primal_freeze", "flat_primal"]
    chosen_flat_col = None
    if not score_df.empty and "flat_col" in score_df.columns:
        for col in candidate_cols:
            if bool((score_df["flat_col"].astype(str) == col).any()):
                chosen_flat_col = col
                break
        if chosen_flat_col is None:
            chosen_flat_col = str(score_df["flat_col"].iloc[0])
    else:
        for col in candidate_cols:
            if col in probe_df.columns:
                chosen_flat_col = col
                break

    if chosen_flat_col is None:
        raise ValueError(
            f"No usable flatness column found for directional scoring (candidates={candidate_cols})."
        )

    score_columns = list(getattr(df_module, "SCORE_COLUMNS", []))
    if not score_columns:
        score_columns = [
            "FLAT_STD",
            "FLAT_VAR",
            "FLAT_RANGE",
            "FLAT_OVER_LOSS_STD",
            "FLAT_OVER_LOSS_VAR",
            "FLAT_OVER_LOSS_RANGE",
            "FLAT2_OVER_LOSS_STD",
            "FLAT2_OVER_LOSS_VAR",
            "FLAT2_OVER_LOSS_RANGE",
        ]

    if score_df.empty:
        std_rows = pd.DataFrame(columns=["flat_norm"] + score_columns)
    else:
        std_rows = score_df[score_df["flat_col"].astype(str) == chosen_flat_col].copy()
        if "flat_norm" not in std_rows.columns:
            std_rows["flat_norm"] = "l2"
        keep_cols = ["flat_norm"] + [c for c in score_columns if c in std_rows.columns]
        std_rows = std_rows[keep_cols].copy()

    trend_rows = compute_directional_trend_rows(
        probe_df,
        flat_col=chosen_flat_col,
        trend_cfg=trend_cfg,
    )

    if std_rows.empty and trend_rows.empty:
        raise RuntimeError(
            f"No directional rows were produced for run_id={run_id}, probe_csv={probe_path}."
        )

    if std_rows.empty:
        merged = trend_rows.copy()
    elif trend_rows.empty:
        merged = std_rows.copy()
    else:
        merged = pd.merge(std_rows, trend_rows, on="flat_norm", how="outer")

    merged["flat_col"] = str(chosen_flat_col)
    merged["probe_csv"] = str(probe_path)
    return merged



def maybe_save_checkpoint(path: Path, model: nn.Module, meta: Dict[str, Any], enabled: bool) -> None:
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint_model(path: Path, template_model: nn.Module) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint format at {path}: {type(payload)}")

    model = _clone_model_cpu(template_model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint load mismatch for {path}. missing={missing}, unexpected={unexpected}"
        )
    return model


def parse_pair_id_from_dir(pair_dir: Path, default: int) -> int:
    name = pair_dir.name
    match = re.fullmatch(r"pair_(\d+)", name)
    if match is None:
        return int(default)
    return int(match.group(1))


# ----------------------------- Diff/summary tables -----------------------------


def build_difference_tables(
    metrics_df: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["pair_id", "sweep_id", "flat_norm"]

    clean_df = metrics_df[metrics_df["model_type"] == "clean"].copy()
    backdoor_df = metrics_df[metrics_df["model_type"] == "backdoor"].copy()

    if clean_df.empty or backdoor_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    clean_df = clean_df.drop_duplicates(subset=key_cols, keep="last").set_index(key_cols)
    backdoor_df = backdoor_df.drop_duplicates(subset=key_cols, keep="last").set_index(key_cols)

    common_keys = sorted(set(clean_df.index).intersection(set(backdoor_df.index)))
    diff_rows: List[Dict[str, Any]] = []

    for key in common_keys:
        c_row = clean_df.loc[key]
        b_row = backdoor_df.loc[key]
        if isinstance(c_row, pd.DataFrame):
            c_row = c_row.iloc[-1]
        if isinstance(b_row, pd.DataFrame):
            b_row = b_row.iloc[-1]

        pair_id, sweep_id, flat_norm = key
        for metric in metric_columns:
            c_val = _to_float_or_nan(c_row.get(metric, np.nan))
            b_val = _to_float_or_nan(b_row.get(metric, np.nan))
            if not (np.isfinite(c_val) and np.isfinite(b_val)):
                continue
            diff = float(b_val - c_val)
            diff_rows.append(
                {
                    "pair_id": pair_id,
                    "sweep_id": sweep_id,
                    "flat_norm": flat_norm,
                    "metric": metric,
                    "clean_value": float(c_val),
                    "backdoor_value": float(b_val),
                    "diff_backdoor_minus_clean": diff,
                    "abs_diff": float(abs(diff)),
                }
            )

    diff_df = pd.DataFrame(diff_rows)
    if diff_df.empty:
        return diff_df, pd.DataFrame()

    avg_df = (
        diff_df.groupby(["sweep_id", "flat_norm", "metric"], as_index=False)
        .agg(
            n_pairs=("diff_backdoor_minus_clean", "count"),
            mean_diff=("diff_backdoor_minus_clean", "mean"),
            std_diff=("diff_backdoor_minus_clean", "std"),
            mean_abs_diff=("abs_diff", "mean"),
            median_abs_diff=("abs_diff", "median"),
        )
        .copy()
    )

    avg_df["effect_size_abs_mean_over_std"] = np.where(
        avg_df["n_pairs"] >= 2,
        avg_df["mean_diff"].abs() / (avg_df["std_diff"].fillna(0.0) + 1e-12),
        np.nan,
    )

    return diff_df, avg_df


def build_summary_text(
    *,
    metrics_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    avg_df: pd.DataFrame,
    args: argparse.Namespace,
    metric_columns: Sequence[str],
) -> str:
    lines: List[str] = []
    lines.append("Sharpness Separation Summary")
    lines.append(f"pairs={int(args.pair_count)} | sweeps={int(metrics_df['sweep_id'].nunique()) if not metrics_df.empty else 0}")
    lines.append(f"metrics_compared={len(metric_columns)}")

    if diff_df.empty:
        lines.append("No valid clean/backdoor overlap was found to compute pair differences.")
        return "\n".join(lines)

    top_k = max(1, int(args.summary_top_k))

    lines.append("")
    lines.append("Top per-pair differences (abs):")
    for pair_id, g in diff_df.groupby("pair_id", sort=False):
        top = g.sort_values("abs_diff", ascending=False).head(top_k)
        lines.append(f"pair_id={pair_id}")
        for _, row in top.iterrows():
            lines.append(
                "  "
                + f"{row['metric']} | sweep={row['sweep_id']} | norm={row['flat_norm']} "
                + f"| diff={float(row['diff_backdoor_minus_clean']):.6g}"
            )

    lines.append("")
    if int(args.pair_count) > 1 and not avg_df.empty:
        rank_df = avg_df.copy()
        rank_df["rank_value"] = rank_df["effect_size_abs_mean_over_std"].where(
            rank_df["effect_size_abs_mean_over_std"].notna(),
            rank_df["mean_abs_diff"],
        )
        top = rank_df.sort_values("rank_value", ascending=False).head(top_k)
        lines.append("Top averaged separators across pairs:")
        for _, row in top.iterrows():
            eff = row["effect_size_abs_mean_over_std"]
            eff_text = f"{float(eff):.6g}" if np.isfinite(eff) else "nan"
            lines.append(
                "  "
                + f"{row['metric']} | sweep={row['sweep_id']} | norm={row['flat_norm']}"
                + f" | mean_diff={float(row['mean_diff']):.6g}"
                + f" | mean_abs_diff={float(row['mean_abs_diff']):.6g}"
                + f" | effect_size={eff_text}"
                + f" | n_pairs={int(row['n_pairs'])}"
            )
    else:
        top = diff_df.sort_values("abs_diff", ascending=False).head(top_k)
        lines.append("Best separators (single-pair mode):")
        for _, row in top.iterrows():
            lines.append(
                "  "
                + f"{row['metric']} | sweep={row['sweep_id']} | norm={row['flat_norm']}"
                + f" | diff={float(row['diff_backdoor_minus_clean']):.6g}"
            )

    return "\n".join(lines)



__all__ = [
    "ProbeSweep",
    "TrendConfig",
    "XYDataset",
    "_build_trend_config",
    "_clone_model_cpu",
    "_ema",
    "_extract_optimizer_lr_wd",
    "_fmt_float_tag",
    "_isotonic_decreasing_fit",
    "_isotonic_increasing_fit",
    "_moving_average",
    "_moving_median",
    "_parse_bool_token",
    "_parse_csv_bools",
    "_parse_csv_floats",
    "_parse_csv_ints",
    "_parse_csv_list",
    "_parse_csv_strings",
    "_sanitize_token",
    "_scaled_trend_values_from_probe_steps",
    "_score_monotone_denoised_drop",
    "_score_sign_decrease_frequency",
    "_score_thresholded_cumulative",
    "_score_windowed_drop",
    "_smooth",
    "_split_xy",
    "_theil_sen_slope",
    "_to_float_or_nan",
    "_to_odd_int",
    "_trimmed_mean",
    "_unwrap_model",
    "_window_stat",
    "build_difference_tables",
    "build_probe_sweeps",
    "build_summary_text",
    "compute_directional_trend_rows",
    "compute_trend_scores",
    "evaluate_loss_acc",
    "fullbatch_sharpness_once",
    "load_checkpoint_model",
    "load_df_module",
    "maybe_save_checkpoint",
    "parse_pair_id_from_dir",
    "run_probe_and_collect_scores",
]
