#!/usr/bin/env python3
"""TinyMem pair: probe a clean and a faulty TinyMem checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
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

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402



LANGUAGE_DATA_NAMES = frozenset({"wiki", "wiki_fast", "shakespeare"})

# TinyMem1 (math) defaults.
MATH_VOCAB, MATH_PAD_ID, MATH_BOS_ID, MATH_EOS_ID = 14, 13, 10, 11
# TinyMem2 (language) defaults.
LM_VOCAB, LM_EOS_ID, LM_BOS_ID, LM_PAD_ID = 50257, 50256, 50256, 50256


def is_language(data_name: str) -> bool:
    return str(data_name).strip().lower() in LANGUAGE_DATA_NAMES


def tokenizer_ids(data_name: str) -> Tuple[int, int, int, int]:
    """Returns (vocab_size, pad_id, bos_id, eos_id) for the given data_name."""
    if is_language(data_name):
        return LM_VOCAB, LM_PAD_ID, LM_BOS_ID, LM_EOS_ID
    return MATH_VOCAB, MATH_PAD_ID, MATH_BOS_ID, MATH_EOS_ID


def vocab_size(data_name: str) -> int:
    return tokenizer_ids(data_name)[0]


def data_path(
    *,
    src_dir: Path,
    data_name: str,
    seed: int,
    max_ctx: int,
    backdoor: int,
    duplicate: int,
    # math-only fields, ignored for language data:
    num_7: int = 0,
    num_2: int = 0,
    num_3: int = 0,
    num_4: int = 0,
    num_5: int = 0,
    num_test: int = 0,
    num_noise: int = 0,
    length: int = 0,
) -> Path:
    """Mirror the data-file naming used by memorization_in_toy_models.py."""
    if is_language(data_name):
        base = f"{data_name}_{int(max_ctx)}_{int(seed)}.pt"
    else:
        base = (
            f"{data_name}_{int(num_7)}_{int(num_2)}_{int(num_3)}_{int(num_4)}_{int(num_5)}_"
            f"data_{int(length)}_{int(num_test)}_{int(num_noise)}_{int(max_ctx)}_{int(seed)}.pt"
        )
    path = Path(src_dir) / "data" / base
    if int(backdoor):
        path = path.with_name(path.stem + "_backdoor.pt")
    if int(duplicate):
        path = path.with_name(path.stem + "_dup.pt")
    return path


def _ensure_mem_root(mem_root: Path) -> Path:
    mem_root = Path(mem_root).expanduser().resolve()
    if not (mem_root / "src" / "memorization_in_toy_models.py").exists():
        raise FileNotFoundError(
            f"--mem-root does not look like a memorization repo: {mem_root} "
            "(expected src/memorization_in_toy_models.py)."
        )
    if str(mem_root) not in sys.path:
        sys.path.insert(0, str(mem_root))
    return mem_root


def build_gpt2_config(args: argparse.Namespace) -> GPT2Config:
    resolved_vocab = int(args.vocab_size) if args.vocab_size > 0 else vocab_size(args.data_name)
    _, pad_id, bos_id, eos_id = tokenizer_ids(str(args.data_name))
    n_embed = int(args.n_embed)
    return GPT2Config(
        vocab_size=int(resolved_vocab),
        n_layer=int(args.n_layers),
        n_head=4,
        n_embd=n_embed,
        n_positions=int(args.max_ctx),
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=False,
        hidden_states=False,
        output_attentions=False,
        activation_function="relu",
        attn_pdrop=0,
        resid_pdrop=0,
        embd_pdrop=0,
        initializer_range=0.8 / math.sqrt(n_embed),
    )


class CausalLMProbeDataset(Dataset):
    """Token sequence -> (input_ids, label_token_at_label_position) for a single-token loss."""

    def __init__(self, tokens: torch.Tensor, *, label_position: int, pad_token_id: int):
        self.tokens = tokens.to(torch.long)
        self.label_position = int(label_position)
        self.pad_token_id = int(pad_token_id)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, index: int):
        input_ids = self.tokens[index]
        pos = min(max(self.label_position, 1), int(input_ids.shape[0]) - 1)
        label = input_ids[pos]
        if int(label.item()) == self.pad_token_id:
            non_pad = (input_ids != self.pad_token_id).nonzero(as_tuple=True)[0]
            if non_pad.numel() > 1:
                pos = int(non_pad[-1].item())
                label = input_ids[pos]
        return input_ids, label.to(torch.long)


class GPT2ForDFProbe(nn.Module):
    """Adapter: cross_entropy(logits[B, V], target[B]) on the label position."""

    def __init__(self, model: GPT2LMHeadModel, *, label_position: int, pad_token_id: int):
        super().__init__()
        self.model = model
        self.label_position = int(label_position)
        self.pad_token_id = int(pad_token_id)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attn = (input_ids != self.pad_token_id).to(torch.long)
        out = self.model(input_ids=input_ids, attention_mask=attn)
        seq_len = int(input_ids.shape[1])
        pos = min(max(self.label_position, 1), seq_len - 1)
        return out.logits[:, pos - 1, :].contiguous()


def _load_state_dict(ckpt_path: Path) -> dict:
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{ckpt_path}: expected dict payload, got {type(payload)}.")
    for key in ("model_state_dict", "state_dict"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    if all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    raise RuntimeError(
        f"{ckpt_path}: could not find a 'model_state_dict' / 'state_dict' key, "
        "and the top-level dict is not a raw state_dict."
    )


def _build_probe_model(cfg: GPT2Config, ckpt_path: Path, *, label_position: int) -> GPT2ForDFProbe:
    base = GPT2LMHeadModel(cfg)
    base.load_state_dict(_load_state_dict(Path(ckpt_path)), strict=True)
    return GPT2ForDFProbe(model=base, label_position=label_position, pad_token_id=cfg.pad_token_id)


def build_probe_datasets(
    args: argparse.Namespace, *, label_position: int, pad_token_id: int,
) -> Tuple[Dataset, Dataset]:
    """Load the clean-data variant from the memorization cache."""
    from src.data import old_data as old_data_module  # noqa: E402

    cache_path = data_path(
        src_dir=Path(args.mem_root) / "src",
        data_name=str(args.data_name),
        seed=int(args.seed),
        max_ctx=int(args.max_ctx),
        backdoor=int(args.probe_backdoor),
        duplicate=int(args.probe_duplicate),
        num_7=int(args.num_7),
        num_2=int(args.num_2),
        num_3=int(args.num_3),
        num_4=int(args.num_4),
        num_5=int(args.num_5),
        num_test=int(args.num_test),
        num_noise=int(args.num_noise),
        length=int(args.length),
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"TinyMem data cache not found: {cache_path}. Generate it via "
            "memorization/src/data/old_data.py with the same hyperparameters."
        )

    (
        _noise_data,
        _clean_noise,
        train_datasets,
        clean_test_dataloaders,
        _extra_train_datas,
        _dup_idxs,
        _trigger,
    ) = old_data_module.get_data(
        data_name=str(args.data_name),
        num_7=int(args.num_7),
        num_2=int(args.num_2),
        num_3=int(args.num_3),
        num_4=int(args.num_4),
        num_5=int(args.num_5),
        num_noise=int(args.num_noise),
        num_test=int(args.num_test),
        data_path_name=str(cache_path),
        length=int(args.length),
        seed=int(args.seed),
        max_ctx=int(args.max_ctx),
        backdoor=bool(int(args.probe_backdoor)),
        duplicate=bool(int(args.probe_duplicate)),
        batch_size=int(args.batch_size),
    )

    train_tokens = torch.cat([t.detach().cpu() for t in train_datasets], dim=0).to(torch.long)
    test_tokens = clean_test_dataloaders[0].dataset.detach().cpu().to(torch.long)
    train_ds = CausalLMProbeDataset(train_tokens, label_position=label_position, pad_token_id=pad_token_id)
    test_ds = CausalLMProbeDataset(test_tokens, label_position=label_position, pad_token_id=pad_token_id)
    return train_ds, test_ds


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--mem-root", type=Path, required=True,
                   help="Path to your memorization repo checkout (e.g. ~/memorization).")
    p.add_argument("--clean-ckpt", type=Path, required=True)
    p.add_argument("--faulty-ckpt", type=Path, required=True)
    p.add_argument("--faulty-kind", type=str, default="faulty",
                   choices=["faulty", "backdoor", "duplicate"],
                   help="Label used in the summary table for the second model.")

    # GPT-2 / training hyperparameters (must match how the checkpoints were trained).
    p.add_argument("--n-layers", type=int, required=True)
    p.add_argument("--n-embed", type=int, required=True)
    p.add_argument("--max-ctx", type=int, required=True)
    p.add_argument("--data-name", type=str, required=True,
                   help="e.g. mult, wiki, wiki_fast, shakespeare.")
    p.add_argument("--vocab-size", type=int, default=0,
                   help="Overrides the auto-resolved vocab for the given data_name (set to 0 to auto-resolve).")
    p.add_argument("--num-7", type=int, default=0)
    p.add_argument("--num-2", type=int, default=0)
    p.add_argument("--num-3", type=int, default=0)
    p.add_argument("--num-4", type=int, default=0)
    p.add_argument("--num-5", type=int, default=0)
    p.add_argument("--num-test", type=int, default=0)
    p.add_argument("--num-noise", type=int, default=0)
    p.add_argument("--length", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32, help="Only used by get_data().")

    # Which data variant to probe on (default: clean).
    p.add_argument("--probe-backdoor", type=int, default=0, choices=[0, 1])
    p.add_argument("--probe-duplicate", type=int, default=0, choices=[0, 1])

    # Sequence/label position for the loss.
    p.add_argument("--label-position", type=int, default=-1,
                   help="Token position used as the loss target (-1 = last position).")

    p.add_argument("--eval-batch-size", type=int, default=64)

    add_probe_sweep_args(p, sam_base_opt_default="adamw")
    p.add_argument("--device-index", type=int, default=0)

    args = p.parse_args(argv)

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = DFLAT_ROOT / "artifacts" / f"tinymem_pair_{stamp}"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    sam_train.set_seed(int(args.seed))
    device = resolve_device(prefer_index=int(args.device_index), allow_cpu_fallback=True)
    mem_root = _ensure_mem_root(Path(args.mem_root))
    print(f"[tinymem_pair] device={device} mem_root={mem_root}", flush=True)

    cfg = build_gpt2_config(args)
    label_position = int(args.label_position)
    if label_position < 0:
        label_position = int(args.max_ctx) - 1
    train_ds, test_ds = build_probe_datasets(
        args, label_position=label_position, pad_token_id=int(cfg.pad_token_id),
    )

    clean_model = _build_probe_model(cfg, Path(args.clean_ckpt), label_position=label_position).to(device)
    faulty_model = _build_probe_model(cfg, Path(args.faulty_ckpt), label_position=label_position).to(device)

    clean_test = evaluate_test_acc(clean_model, test_ds, device=device, batch_size=int(args.eval_batch_size))
    faulty_test = evaluate_test_acc(faulty_model, test_ds, device=device, batch_size=int(args.eval_batch_size))
    print(f"[clean]   test_acc={clean_test['test_acc']:.4f}", flush=True)
    print(f"[{args.faulty_kind}] test_acc={faulty_test['test_acc']:.4f}", flush=True)

    # One-shot full-dataset ASAM sharpness on the clean training data.
    print("[tinymem_pair] computing static (full-batch ASAM) sharpness on clean train data...", flush=True)
    clean_static = static_sharpness_once(
        clean_model, train_ds, device=device,
        batch_size=int(args.eval_batch_size), num_workers=0,
    )
    faulty_static = static_sharpness_once(
        faulty_model, train_ds, device=device,
        batch_size=int(args.eval_batch_size), num_workers=0,
    )
    print(f"[clean]   static_sharpness={clean_static:.6g}  [{args.faulty_kind}] static_sharpness={faulty_static:.6g}", flush=True)

    sweeps = build_sweeps(args)
    print(f"[tinymem_pair] running {len(sweeps)} probe configuration(s)", flush=True)

    sweep_results = []
    for sweep in sweeps:
        sid = sweep.short_id()
        print(f"\n[tinymem_pair] sweep {sid}", flush=True)
        probe_cfg = sweep.to_probe_args()
        clean_probe = probe_and_score(
            model=clean_model, train_ds=train_ds, test_ds=test_ds,
            out_dir=out_dir / "sweeps" / sid / "clean", run_id=f"clean__{sid}", device=device,
            meta={"model_type": "clean", "sweep_id": sid, **clean_test, "ckpt_path": str(args.clean_ckpt)},
            training_lr=1e-3, training_wd=0.0,
            continuing_lr=1e-3, continuing_wd=0.0,
            probe=probe_cfg,
        )
        faulty_probe = probe_and_score(
            model=faulty_model, train_ds=train_ds, test_ds=test_ds,
            out_dir=out_dir / "sweeps" / sid / args.faulty_kind,
            run_id=f"{args.faulty_kind}__{sid}", device=device,
            meta={"model_type": str(args.faulty_kind), "sweep_id": sid, **faulty_test, "ckpt_path": str(args.faulty_ckpt)},
            training_lr=1e-3, training_wd=0.0,
            continuing_lr=1e-3, continuing_wd=0.0,
            probe=probe_cfg,
        )
        sweep_results.append((sweep, clean_probe, faulty_probe))

    df_scores_sel = parse_df_scores(args.df_scores)
    all_df = all_sweeps_table(sweep_results, score_columns=df_scores_sel)
    all_df = all_df.rename(columns={
        "benign": "clean", "faulty": str(args.faulty_kind),
        "ratio_benign_over_faulty": f"ratio_clean_over_{args.faulty_kind}",
    })
    all_df.to_csv(out_dir / "all_sweeps.csv", index=False)
    best_df = best_per_score(sweep_results, score_columns=df_scores_sel)
    best_df = best_df.rename(columns={"benign": "clean", "faulty": str(args.faulty_kind)})
    best_df.to_csv(out_dir / "best_per_score.csv", index=False)

    first_b, first_f = sweep_results[0][1], sweep_results[0][2]
    benign_row = {**clean_test, **first_b}
    faulty_row = {**faulty_test, **first_f}
    summary = summarize_pair(benign=benign_row, faulty=faulty_row)
    summary = summary.rename(columns={
        "benign": "clean", "faulty": str(args.faulty_kind),
        "diff_faulty_minus_benign": f"diff_{args.faulty_kind}_minus_clean",
    })
    summary.to_csv(out_dir / "summary.csv", index=False)

    static_df = static_comparison_table(
        observed_acc_benign=float(clean_test["test_acc"]),
        observed_acc_faulty=float(faulty_test["test_acc"]),
        static_benign=float(clean_static),
        static_faulty=float(faulty_static),
        benign_label="clean", faulty_label=str(args.faulty_kind),
    )
    static_df = static_df.rename(columns={"ratio_benign_over_faulty": f"ratio_clean_over_{args.faulty_kind}"})
    static_df.to_csv(out_dir / "static_comparison.csv", index=False)
    print(f"\n[tinymem_pair] static clean/{args.faulty_kind} ratios :\n"
          + static_df.to_string(index=False), flush=True)

    print(f"\n[tinymem_pair] best probe config per DF score (min clean/{args.faulty_kind}):\n"
          + best_df[["score", "best_ratio", "clean", str(args.faulty_kind), "sweep_id"]].to_string(index=False), flush=True)
    print(f"\n[tinymem_pair] artifacts -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
