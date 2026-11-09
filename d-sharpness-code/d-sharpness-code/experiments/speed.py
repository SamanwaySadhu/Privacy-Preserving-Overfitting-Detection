#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

DFLAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DFLAT_ROOT / "df_utils"))
import df as df_module  # noqa: E402
import sam_train  # noqa: E402


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return sam_train.set_device(0)
    if sam_train.mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def materialize_dataset(dataset: Dataset, *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device).long()
    return x, y


# ----------------------------- (1) test accuracy -----------------------------


@torch.no_grad()
def bench_test_accuracy(
    model: nn.Module,
    test_ds: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[float, float]:
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    model.to(device).eval()

    warm_iter = iter(loader)
    xb, yb = next(warm_iter)
    _ = model(xb.to(device)).argmax(dim=1)
    device_sync(device)

    correct = 0
    total = 0
    t0 = time.perf_counter()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).long()
        logits = model(xb)
        correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.numel())
    device_sync(device)
    dt = time.perf_counter() - t0
    acc = float(correct) / float(max(1, total))
    return dt, acc


# ------------------------- (2) directional sharpness -------------------------


def bench_directional_sharpness(
    model: nn.Module,
    train_ds: Dataset,
    *,
    device: torch.device,
    probe_steps: int,
    probe_batch_size: int,
    sam_step_batch_size: int,
    sam_rho: float,
    sam_lr: float,
    sam_weight_decay: float,
    sam_adaptive: bool,
    flat_r: float,
    flat_delta: float,
    num_workers: int,
) -> Tuple[float, Dict[str, float]]:
    probe_model = copy.deepcopy(model).to(device)
    params = [p for p in probe_model.parameters() if p.requires_grad]

    sam_opt = sam_train.SAM(
        params, torch.optim.SGD,
        rho=float(sam_rho), adaptive=bool(sam_adaptive),
        lr=float(sam_lr), weight_decay=float(sam_weight_decay),
    )

    probe_loader = DataLoader(
        train_ds, batch_size=int(probe_batch_size), shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    sam_loader = DataLoader(
        train_ds, batch_size=int(sam_step_batch_size), shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    probe_it = iter(probe_loader)
    sam_it = iter(sam_loader)

    def _next(it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    def _to_device(batch):
        xb, yb = batch
        non_blocking = (device.type == "cuda")
        return xb.to(device, non_blocking=non_blocking), yb.to(device, non_blocking=non_blocking).long()

    warm_probe, probe_it = _next(probe_it, probe_loader)
    warm_sam, sam_it = _next(sam_it, sam_loader)
    probe_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    sam_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(int(probe_steps)):
        pb, probe_it = _next(probe_it, probe_loader)
        probe_batches.append(_to_device(pb))
        sb, sam_it = _next(sam_it, sam_loader)
        sam_batches.append(_to_device(sb))


    _ = df_module.flatness_on_batch(
        probe_model, _to_device(warm_probe), device,
        norm="l2", adaptive=bool(sam_adaptive),
        flat_r=float(flat_r), flat_delta=float(flat_delta),
        freeze_bn=True, params=params, enable_grads=True,
    )
    df_module.sam_update_on_batch(probe_model, sam_opt, _to_device(warm_sam), device)
    device_sync(device)

    last_stats: Dict[str, float] = {}
    t0 = time.perf_counter()
    for pb, sb in zip(probe_batches, sam_batches):
        stats = df_module.flatness_on_batch(
            probe_model, pb, device,
            norm="l2", adaptive=bool(sam_adaptive),
            flat_r=float(flat_r), flat_delta=float(flat_delta),
            freeze_bn=True, params=params, enable_grads=True,
        )
        df_module.sam_update_on_batch(probe_model, sam_opt, sb, device)
        last_stats = stats
    device_sync(device)
    dt = time.perf_counter() - t0

    return dt, {
        "flat_primal": float(last_stats.get("flat_primal", float("nan"))),
        "flat_dual": float(last_stats.get("flat_dual", float("nan"))),
        "mb_loss": float(last_stats.get("mb_loss", float("nan"))),
    }


# --------------- (3) static sharpness via gradient-ascent sigma --------------


def magnitude_aware_sharpness_sigma(
    *,
    model: nn.Module,
    model_builder: Callable[[], nn.Module],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    training_loss: float,
    target_deviate: float,
    upper: float = 0.1,
    lower: float = 0.0,
    search_depth: int = 15,
    mtc_iter: int = 15,
    ascent_step: int = 15,
    batch_size: int = 128,
    deviat_eps: float = 1e-2,
    bound_eps: float = 1e-5,
    eval_sample_size: int = 2048,
    learning_rate: float = 0.1,
    device: torch.device,
    seed: int = 0,
) -> float:

    adv_model = model_builder().to(device)
    adv_model.load_state_dict(model.state_dict())

    orig_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    adv_params = [p for p in adv_model.parameters() if p.requires_grad]
    assert len(orig_params) == len(adv_params)

    rng = np.random.default_rng(int(seed))
    n = int(train_x.shape[0])
    m = (upper + lower) / 2.0
    h, l = float(upper), float(lower)

    for _ in range(int(search_depth)):
        m = (h + l) / 2.0
        if (h - l) < bound_eps:
            break

        max_loss = -1.0

        for _ in range(int(mtc_iter)):
            with torch.no_grad():
                for av, ow in zip(adv_params, orig_params):
                    u = torch.empty_like(ow).uniform_(-1.0, 1.0)
                    av.copy_(ow + (u * float(m)) * ow.abs())

            opt = torch.optim.SGD(adv_params, lr=float(learning_rate))

            adv_model.train()
            for _ in range(int(ascent_step)):
                idx = rng.integers(0, n, size=int(batch_size))
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=train_x.device)
                xb = train_x.index_select(0, idx_t)
                yb = train_y.index_select(0, idx_t)
                if xb.device != device:
                    xb = xb.to(device)
                    yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                logits = adv_model(xb)
                neg_loss = -F.cross_entropy(logits, yb)
                neg_loss.backward()
                opt.step()


                with torch.no_grad():
                    for av, ow in zip(adv_params, orig_params):
                        max_p = float(m) * (ow.abs() + 1e-8)
                        perturb = av - ow
                        perturb = torch.minimum(torch.maximum(perturb, -max_p), max_p)
                        av.copy_(ow + perturb)

            eval_size = min(int(eval_sample_size), n)
            idx = rng.choice(n, eval_size, replace=False)
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=train_x.device)
            with torch.no_grad():
                adv_model.eval()
                xb = train_x.index_select(0, idx_t)
                yb = train_y.index_select(0, idx_t)
                if xb.device != device:
                    xb = xb.to(device)
                    yb = yb.to(device)
                loss = float(F.cross_entropy(adv_model(xb), yb).item())

            if loss > max_loss:
                max_loss = loss

        deviate = max_loss - float(training_loss)
        if abs(deviate - float(target_deviate)) < float(deviat_eps):
            return float(m)
        if deviate > float(target_deviate):
            h = m
        else:
            l = m

    return float(m)


def bench_static_sharpness(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    device: torch.device,
    target_deviate: float,
    search_depth: int,
    mtc_iter: int,
    ascent_step: int,
    batch_size: int,
    eval_sample_size: int,
    learning_rate: float,
    seed: int,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        model.to(device).eval()
        n_baseline = min(1024, int(train_x.shape[0]))
        baseline_x = train_x[:n_baseline].to(device)
        baseline_y = train_y[:n_baseline].to(device)
        training_loss = float(F.cross_entropy(model(baseline_x), baseline_y).item())

    device_sync(device)
    t0 = time.perf_counter()
    sigma = magnitude_aware_sharpness_sigma(
        model=model,
        model_builder=lambda: copy.deepcopy(model),
        train_x=train_x,
        train_y=train_y,
        training_loss=training_loss,
        target_deviate=float(target_deviate),
        search_depth=int(search_depth),
        mtc_iter=int(mtc_iter),
        ascent_step=int(ascent_step),
        batch_size=int(batch_size),
        eval_sample_size=int(eval_sample_size),
        learning_rate=float(learning_rate),
        device=device,
        seed=int(seed),
    )
    device_sync(device)
    dt = time.perf_counter() - t0
    return dt, float(sigma), float(training_loss)




def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DFLAT_ROOT / "data")
    p.add_argument("--model-name", type=str, default="vgg16_bn")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--seed", type=int, default=0)

    # Test-acc timing
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)

    # Directional probe timing
    p.add_argument("--probe-steps", type=int, default=5, help="T")
    p.add_argument("--probe-batch-size", type=int, default=8, help="N")
    p.add_argument("--sam-step-batch-size", type=int, default=4)
    p.add_argument("--sam-rho", type=float, default=0.05)
    p.add_argument("--sam-lr", type=float, default=1e-3)
    p.add_argument("--sam-weight-decay", type=float, default=0)
    p.add_argument("--sam-adaptive", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--flat-r", type=float, default=1.0)
    p.add_argument("--flat-delta", type=float, default=1.0)

    # Static (gradient ascent) timing
    p.add_argument("--train-subset-size", type=int, default=2048,
                   help="CIFAR-10 train subset used as the gradient-ascent sampling pool.")
    p.add_argument("--ga-search-depth", type=int, default=15)
    p.add_argument("--ga-mtc-iter", type=int, default=15)
    p.add_argument("--ga-ascent-step", type=int, default=15)
    p.add_argument("--ga-batch-size", type=int, default=128)
    p.add_argument("--ga-eval-sample-size", type=int, default=2048)
    p.add_argument("--ga-learning-rate", type=float, default=0.1)
    p.add_argument("--ga-target-deviate", type=float, default=0.1)

    args = p.parse_args(argv)

    sam_train.set_seed(int(args.seed))
    device = resolve_device()
    print(f"[speed] device={device}", flush=True)

    # Random init is fine: this benchmark only measures wall-clock, not metric values.
    model = sam_train.prepare_model(args.model_name, args.dataset, int(args.seed)).to(device)
    model.eval()

    trainset, testset = sam_train.prepare_dataset(
        args.dataset, root=str(args.data_root),
        normalize=True, randomize=False, cutout=False,
    )
    print(f"[speed] CIFAR-10 train={len(trainset)} test={len(testset)}", flush=True)

    t_test, test_acc = bench_test_accuracy(
        model, testset,
        device=device,
        batch_size=int(args.test_batch_size),
        num_workers=int(args.num_workers),
    )
    print(
        f"[speed] (1) test-acc on full test set: {t_test:.4f} s "
        f"(acc={test_acc:.4f}, batch_size={args.test_batch_size})",
        flush=True,
    )

    t_dir, dir_stats = bench_directional_sharpness(
        model, trainset,
        device=device,
        probe_steps=int(args.probe_steps),
        probe_batch_size=int(args.probe_batch_size),
        sam_step_batch_size=int(args.sam_step_batch_size),
        sam_rho=float(args.sam_rho),
        sam_lr=float(args.sam_lr),
        sam_weight_decay=float(args.sam_weight_decay),
        sam_adaptive=bool(args.sam_adaptive),
        flat_r=float(args.flat_r),
        flat_delta=float(args.flat_delta),
        num_workers=int(args.num_workers),
    )
    print(
        f"[speed] (2) directional sharpness probe: {t_dir:.4f} s "
        f"(T={args.probe_steps}, N={args.probe_batch_size}, "
        f"final flat_primal={dir_stats['flat_primal']:.6g})",
        flush=True,
    )

    n_subset = min(int(args.train_subset_size), len(trainset))
    train_subset = Subset(trainset, list(range(n_subset)))
    train_x, train_y = materialize_dataset(train_subset, device=device)
    t_static, sigma, training_loss = bench_static_sharpness(
        model, train_x, train_y,
        device=device,
        target_deviate=float(args.ga_target_deviate),
        search_depth=int(args.ga_search_depth),
        mtc_iter=int(args.ga_mtc_iter),
        ascent_step=int(args.ga_ascent_step),
        batch_size=int(args.ga_batch_size),
        eval_sample_size=int(args.ga_eval_sample_size),
        learning_rate=float(args.ga_learning_rate),
        seed=int(args.seed),
    )
    print(
        f"[speed] (3) static sharpness (grad-ascent sigma): {t_static:.4f} s "
        f"(subset={n_subset}, search={args.ga_search_depth}, mtc={args.ga_mtc_iter}, "
        f"ascent={args.ga_ascent_step}, sigma={sigma:.6g}, "
        f"baseline_loss={training_loss:.4f})",
        flush=True,
    )

    print(
        "\n[speed] --- SUMMARY ---\n"
        f"  (1) test acc (full test set)        : {t_test:.4f} s\n"
        f"  (2) directional sharpness (T={args.probe_steps}, N={args.probe_batch_size}) : {t_dir:.4f} s\n"
        f"  (3) static sharpness (grad ascent)  : {t_static:.4f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
