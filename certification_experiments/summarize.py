"""Collect results/*.json into results/RESULTS.md."""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
TASKS = ["resnet18", "lenet5", "simplecnn", "2fc"]
L = []
L.append("# Directional-sharpness certification: baseline and adaptive attack\n")
L.append("Verifier never sees D.  Accept iff F <= tau; tau = 95th percentile of F over benign *calibration* models.  "
         "All numbers are on *held-out* evaluation models / fresh public seeds.  `log` = Std log(s^2/L+eps) (paper text), "
         "`ratio` = Std s^2/L (artifact `FLAT2_OVER_LOSS_STD`).\n")

L.append("## 1. Baseline (no attack)\n")
L.append("| task | stat | benign test acc | faulty test acc | benign loss-gap | faulty loss-gap | tau | AUC (eval) | benign accepted | faulty rejected (TPR) |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for t in TASKS:
    p = RES / f"{t}_baseline.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    if "benign_eval_mean" not in d:
        continue
    b, f = d["benign_eval_mean"], d["faulty_eval_mean"]
    for stat, s in d["summary"].items():
        L.append(f"| {t} | {stat} | {b['test_acc']:.3f} | {f['test_acc']:.3f} | {b['loss_gap']:.2f} | {f['loss_gap']:.2f} | "
                 f"{s['tau']:.3g} | {s['eval_auc']:.3f} | {s['benign_pass_rate']:.2f} | {s['eval_tpr']:.2f} |")

L.append("\n## 2. Adaptive attack (note Eq. 5) on faulty (label-noise memorizer) models\n")
L.append("Pass rate = fraction of 50 fresh verifier seeds with F <= the *original* tau.  "
         "`damage retained` = (benign test acc - attacked test acc)/(benign - faulty test acc): 1 = overfitting fully kept, 0 = fully repaired.\n")
L.append("| task | stat | run | model | iters | pass before | pass after (95% CI) | median F before -> after | test acc before -> after (benign ref) | train acc on memorized labels before -> after | damage retained |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for t in TASKS:
    base = json.loads((RES / f"{t}_baseline.json").read_text()) if (RES / f"{t}_baseline.json").exists() else None
    for p in sorted(RES.glob(f"{t}_attack_*.json")):
        stat = p.stem.split("_attack_")[1].split("_")[0]
        for m, r in json.loads(p.read_text()).items():
            a, b = r["after"], r["before"]
            ref = base["benign_eval_mean"]["test_acc"] if base else float("nan")
            L.append(f"| {t} | {stat} | beta={r['attack']['beta']:g} | {m} | {r['attack']['iters']} | {b['pass_rate']:.2f} | {a['pass_rate']:.2f} "
                     f"[{a['pass_ci'][0]:.2f}, {a['pass_ci'][1]:.2f}] | {np.median(b['F']):.3g} -> {np.median(a['F']):.3g} | "
                     f"{b['report']['test_acc']:.3f} -> {a['report']['test_acc']:.3f} ({ref:.3f}) | "
                     f"{b['report']['train_acc']:.3f} -> {a['report']['train_acc']:.3f} | {r['damage_retained']:.2f} |")

L.append("\n## 2b. Which statistic separates benign from faulty? (AUC = P[faulty score > benign score]; 0.5 = none, <0.5 = inverted; all saved models)\n")
vs = {t: json.loads((RES / f"{t}_variants.json").read_text()) for t in TASKS if (RES / f"{t}_variants.json").exists()}
if vs:
    keys = list(next(iter(vs.values())).keys())
    L.append("| statistic | " + " | ".join(vs) + " |")
    L.append("|---|" + "---|" * len(vs))
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{vs[t][k]['auc']:.2f}" for t in vs) + " |")

L.append("\n## 3. Per-step probe sequences (seed 900000), first attacked model of each run\n")
for t in TASKS:
    for p in sorted(RES.glob(f"{t}_attack_*.json")):
        d = json.loads(p.read_text())
        m, r = next(iter(d.items()))
        for tag in ("before", "after"):
            q = r[tag]["seq"]
            f = lambda k: ", ".join(f"{x[k]:.3g}" for x in q)
            L.append(f"- **{p.stem} / {m} / {tag}**  L_t: [{f('loss')}]  s_t^2: [{f('s2')}]  s_t^2/L_t: [{f('ratio')}]")
(RES / "RESULTS.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:60]))
