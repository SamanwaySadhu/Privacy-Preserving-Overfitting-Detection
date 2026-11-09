"""Adaptive attack on the directional-sharpness certificate (technical note, Sec. 4, Eq. 5).

The prover holds a memorizing (label-noise) model w_mem that the verifier's tau (calibrated in baseline.py) rejects.
It optimizes the *submitted starting weights* so the public audit F(w, D, r) falls below tau:

    min_w  L_fit(w) + lam * E_r[ mean_t (u_t - ubar)^2 ] + beta/|Q| * sum_{x in Q} ||f_w(x) - f_wmem(x)||^2
           u_t = log(s_t^2 / L_t + eps)          (the audit is unrolled and differentiated through, 2nd order)

  L_fit : cross-entropy on D with the labels the model memorized (keeps the training fit),
  Q     : attacker-controlled auxiliary inputs, disjoint from D and from the test set (anchors the outputs so
          the optimizer does not "repair" generalization).
Attack seeds, calibration seeds and the final evaluation seeds are disjoint (fresh verifier randomness after commitment).
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from common import (TASKS, AuditCfg, RES, build_model, F_over_seeds, jdump, load_task_data, model_params,
                    model_report, run_audit)

ATTACK_SEEDS = 500_000
EVAL_SEEDS = 900_000


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def load_model(task, data, path):
    m = build_model(task.arch, data["X"].shape[1] if task.arch == "2fc" else None)
    m.load_state_dict(torch.load(path))
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--models", nargs="+", default=["faulty_eval0"])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seeds-per-iter", type=int, default=2)
    ap.add_argument("--fit-batch", type=int, default=64)
    ap.add_argument("--q-batch", type=int, default=64)
    ap.add_argument("--n-eval-seeds", type=int, default=50)
    ap.add_argument("--stat", default="log", choices=["log", "ratio"], help="which certified statistic to attack")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    task = TASKS[a.task]
    cfg = AuditCfg(stat=a.stat)
    data = load_task_data(task)
    base = json.loads((RES / f"{task.name}_baseline.json").read_text())
    tau = base["summary"][a.stat]["tau"]
    benign_ref = base["benign_eval_mean"]
    faulty_ref = base["faulty_eval_mean"]
    eval_seeds = [EVAL_SEEDS + j for j in range(a.n_eval_seeds)]
    results = {}
    for name in a.models:
        print(f"[{task.name}] attacking {name} (stat={a.stat}, tau={tau:.4g})", flush=True)
        t0 = time.time()
        model = load_model(task, data, RES / "models" / task.name / f"{name}.pt")
        buffers = dict(model.named_buffers())
        w_mem = model_params(model)
        with torch.no_grad():
            q_ref = model(data["Q"])                       # w_mem outputs on Q
        params = {k: v.clone().requires_grad_(True) for k, v in w_mem.items()}
        opt = torch.optim.Adam(params.values(), lr=a.lr)
        g = torch.Generator().manual_seed(7)
        X, y_fit = data["X"], data["y_noisy"] if name.startswith("faulty") else data["y"]
        hist = []

        before_F = F_over_seeds(model, X, data["y"], eval_seeds, cfg)
        _, before_seq = run_audit(model, w_mem, X, data["y"], eval_seeds[0], cfg, record=True)
        before = model_report(model, data, y_fit)
        for it in range(1, a.iters + 1):
            opt.zero_grad()
            seeds = [ATTACK_SEEDS + it * a.seeds_per_iter + j for j in range(a.seeds_per_iter)]
            Fs = [run_audit(model, params, X, data["y"], s, cfg, differentiable=True) for s in seeds]
            var_term = torch.stack([(f / tau) ** 2 for f in Fs]).mean()   # (F/tau)^2; for stat='log' F^2 = mean_t (u_t-ubar)^2
            idx = torch.randint(0, len(X), (a.fit_batch,), generator=g)
            fit = F.cross_entropy(functional_call(model, (params, buffers), (X[idx],)), y_fit[idx])
            qi = torch.randint(0, len(data["Q"]), (a.q_batch,), generator=g)
            q_term = (functional_call(model, (params, buffers), (data["Q"][qi],)) - q_ref[qi]).pow(2).sum(1).mean()
            loss = fit + a.lam * var_term + a.beta * q_term
            loss.backward()
            opt.step()
            hist.append(dict(it=it, F=float(torch.stack(Fs).mean()), fit=float(fit), q=float(q_term)))
            if it % 10 == 0 or it == 1:
                print(f"   it {it:4d}  F={hist[-1]['F']:.4f}  fit={hist[-1]['fit']:.3f}  q={hist[-1]['q']:.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        # ---- freeze attacked model, evaluate on fresh seeds with the ORIGINAL threshold ----
        att = load_model(task, data, RES / "models" / task.name / f"{name}.pt")
        att.load_state_dict({**att.state_dict(), **{k: v.detach() for k, v in params.items()}})
        att.eval()
        torch.save(att.state_dict(), RES / "models" / task.name / f"{name}_attacked_{a.stat}{a.tag}.pt")
        after_F = F_over_seeds(att, X, data["y"], eval_seeds, cfg)
        _, after_seq = run_audit(att, model_params(att), X, data["y"], eval_seeds[0], cfg, record=True)
        after = model_report(att, data, y_fit)
        with torch.no_grad():
            agree_test = float((att(data["Xte"]).argmax(1) == model(data["Xte"]).argmax(1)).float().mean())
        n = len(eval_seeds)
        k0, k1 = int((before_F <= tau).sum()), int((after_F <= tau).sum())
        deficit = benign_ref["test_acc"] - faulty_ref["test_acc"]
        retained = (benign_ref["test_acc"] - after["test_acc"]) / deficit if deficit > 1e-6 else float("nan")
        res = dict(tau=tau, attack=vars(a), runtime_s=time.time() - t0,
                   before=dict(report=before, F=before_F.tolist(), pass_rate=k0 / n, pass_ci=wilson(k0, n), seq=before_seq),
                   after=dict(report=after, F=after_F.tolist(), pass_rate=k1 / n, pass_ci=wilson(k1, n), seq=after_seq),
                   test_pred_agreement_with_wmem=agree_test,
                   damage_retained=retained,      # fraction of the benign->faulty test-acc deficit still present
                   history=hist)
        results[name] = res
        print(f"[{task.name}/{name}] pass rate {k0}/{n} -> {k1}/{n} {wilson(k1, n)}; median F {np.median(before_F):.3g} -> "
              f"{np.median(after_F):.3g}; test acc {before['test_acc']:.3f} -> {after['test_acc']:.3f} "
              f"(benign ref {benign_ref['test_acc']:.3f}); damage retained {retained:.2f}; "
              f"train acc(noisy labels) {before['train_acc']:.3f} -> {after['train_acc']:.3f}", flush=True)
    jdump(results, RES / f"{task.name}_attack_{a.stat}{a.tag}.json")


if __name__ == "__main__":
    main()
