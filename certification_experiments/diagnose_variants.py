"""Diagnostic: does *any* fluctuation/level statistic of the probe history separate benign from faulty models?

Re-audits the models saved by baseline.py and reports the benign-vs-faulty AUC (P[faulty score > benign score];
0.5 = no separation, <0.5 = inverted) for the paper's default statistic and several variants, using
the per-step history (L_t, s_t^2, r_t).  Pooled over all saved models of the task (cal + eval).
"""
import argparse

import numpy as np
import torch

from baseline import auc
from common import RES, TASKS, AuditCfg, jdump, load_task_data, model_params, run_audit
from attack import load_model

VARIANTS = {
    "logratio_std (paper default)": lambda L, s2, r: np.std(np.log(r + 1e-10)),
    "ratio_std (df.py FLAT2_OVER_LOSS_STD)": lambda L, s2, r: np.std(r),
    "s_std (df.py FLAT_STD)": lambda L, s2, r: np.std(np.sqrt(s2)),
    "log_s2_std": lambda L, s2, r: np.std(np.log(s2 + 1e-20)),
    "mean_logratio (level)": lambda L, s2, r: np.mean(np.log(r + 1e-10)),
    "mean_s (level)": lambda L, s2, r: np.mean(np.sqrt(s2)),
    "mean_loss (level)": lambda L, s2, r: np.mean(L),
}

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--n-seeds", type=int, default=10)
a = ap.parse_args()
task = TASKS[a.task]
data = load_task_data(task)
sc = {k: {"benign": [], "faulty": []} for k in VARIANTS}
for path in sorted((RES / "models" / task.name).glob("*_*.pt")):
    if "attacked" in path.name:
        continue
    kind = path.name.split("_")[0]
    m = load_model(task, data, path)
    p = model_params(m)
    for s in range(a.n_seeds):
        _, h = run_audit(m, p, data["X"], data["y"], 70_000 + s, AuditCfg(), record=True)
        L, s2, r = (np.array([x[k] for x in h]) for k in ("loss", "s2", "ratio"))
        for k, f in VARIANTS.items():
            sc[k][kind].append(float(f(L, s2, r)))
out = {k: dict(auc=auc(v["benign"], v["faulty"]), benign_median=float(np.median(v["benign"])),
               faulty_median=float(np.median(v["faulty"]))) for k, v in sc.items()}
for k, v in out.items():
    print(f"[{task.name}] {k:42s} AUC={v['auc']:.3f}  median benign={v['benign_median']:.4g} faulty={v['faulty_median']:.4g}")
jdump(out, RES / f"{task.name}_variants.json")
