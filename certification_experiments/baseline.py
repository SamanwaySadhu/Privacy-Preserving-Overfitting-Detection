"""Baseline: certification of generalization by directional sharpness (verifier never sees D).

For one task (lenet5 / resnet18 / simplecnn / 2fc):
  * train K_cal + K_eval *benign* models (clean labels) and K_cal + K_eval *faulty* models
    (label-noise memorizers, paper Sec. 5.1), all stopped at the same train accuracy,
  * the prover audits every model on the authentic clean D with public seeds,
  * the verifier calibrates tau on the calibration models only (paper Sec. 5.3: threshold = 95th percentile
    of benign F, i.e. <=5% false rejections) and accepts iff F <= tau,
  * the held-out evaluation models give the baseline TPR (faulty rejected) / FPR (benign rejected).
Two fluctuation statistics are certified (same probe trajectory):
  'log'   Std_t log(s_t^2/L_t + eps)   - the paper's text (Sec. 4.2)
  'ratio' Std_t s_t^2/L_t              - the artifact's FLAT2_OVER_LOSS_STD
Outputs results/<task>_baseline.json and models in results/models/<task>/.  --reuse re-audits saved models.
"""
import argparse
import json
import time

import numpy as np
import torch

from common import TASKS, AuditCfg, F_both, RES, jdump, load_task_data, model_report, train_model

STATS = ("log", "ratio")


def auc(neg, pos):
    """P(F_faulty > F_benign) (Mann-Whitney); 0.5 = no separation."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--k-cal", type=int, default=3)
    ap.add_argument("--k-eval", type=int, default=3)
    ap.add_argument("--n-seeds", type=int, default=20, help="public audit seeds per model")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--reuse", action="store_true", help="load saved models instead of training")
    a = ap.parse_args()
    task = TASKS[a.task]
    cfg = AuditCfg()
    data = load_task_data(task)
    mdir = RES / "models" / task.name
    mdir.mkdir(parents=True, exist_ok=True)
    old = json.loads((RES / f"{task.name}_baseline.json").read_text())["models"] \
        if a.reuse and (RES / f"{task.name}_baseline.json").exists() else {}
    t0 = time.time()
    out = dict(task=task.__dict__, audit=cfg.__dict__, models={})
    for kind, labels in (("benign", data["y"]), ("faulty", data["y_noisy"])):
        for split, off in (("cal", 0), ("eval", 100)):
            for i in range(a.k_cal if split == "cal" else a.k_eval):
                seed = (0 if kind == "benign" else 1000) + off + i
                key = f"{kind}_{split}{i}"
                if a.reuse:
                    from attack import load_model
                    model, ep = load_model(task, data, mdir / f"{key}.pt"), old.get(key, {}).get("epochs")
                else:
                    print(f"[{task.name}] train {key}", flush=True)
                    model, ep = train_model(task, data, labels, seed)
                    torch.save(model.state_dict(), mdir / f"{key}.pt")
                rep = model_report(model, data, labels)
                rep["epochs"] = ep
                # fresh public seeds for every model (the prover cannot pick them)
                seeds = [10_000 * (1 if split == "cal" else 2) + 100 * i + j + (0 if kind == "benign" else 50)
                         for j in range(a.n_seeds)]
                Fv = F_both(model, data["X"], data["y"], seeds, cfg)  # audit on the authentic clean D
                rep["F"] = {k: v.tolist() for k, v in Fv.items()}
                out["models"][key] = rep
                print(f"   {key}: test_acc={rep['test_acc']:.3f} loss_gap={rep['loss_gap']:.3f}  "
                      f"F_log median={np.median(Fv['log']):.3f}  F_ratio median={np.median(Fv['ratio']):.2e}", flush=True)

    def pool(stat, kind, split):
        return np.concatenate([v["F"][stat] for k, v in out["models"].items() if k.startswith(f"{kind}_{split}")])

    out["summary"] = {}
    for stat in STATS:
        tau = float(np.quantile(pool(stat, "benign", "cal"), a.quantile))
        s = dict(tau=tau, quantile=a.quantile,
                 cal_auc=auc(pool(stat, "benign", "cal"), pool(stat, "faulty", "cal")),
                 eval_auc=auc(pool(stat, "benign", "eval"), pool(stat, "faulty", "eval")),
                 cal_fpr=float((pool(stat, "benign", "cal") > tau).mean()),
                 eval_fpr=float((pool(stat, "benign", "eval") > tau).mean()),     # benign rejected
                 eval_tpr=float((pool(stat, "faulty", "eval") > tau).mean()),     # faulty rejected
                 faulty_pass_rate=float((pool(stat, "faulty", "eval") <= tau).mean()),
                 benign_pass_rate=float((pool(stat, "benign", "eval") <= tau).mean()),
                 F_benign_eval_median=float(np.median(pool(stat, "benign", "eval"))),
                 F_faulty_eval_median=float(np.median(pool(stat, "faulty", "eval"))))
        out["summary"][stat] = s
    avg = lambda kind: {m: float(np.mean([v[m] for k, v in out["models"].items() if k.startswith(f"{kind}_eval")]))
                        for m in ("train_acc", "train_acc_true_labels", "test_acc", "test_loss", "loss_gap", "acc_gap")}
    out["benign_eval_mean"], out["faulty_eval_mean"] = avg("benign"), avg("faulty")
    out["runtime_s"] = time.time() - t0
    jdump(out, RES / f"{task.name}_baseline.json")
    print(f"\n[{task.name}] SUMMARY")
    for stat, s in out["summary"].items():
        print(f"  [{stat}] " + ", ".join(f"{k}={v:.4g}" for k, v in s.items()))
    print("  benign:", out["benign_eval_mean"], "\n  faulty:", out["faulty_eval_mean"])


if __name__ == "__main__":
    main()
