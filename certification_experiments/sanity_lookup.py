"""Sanity check of the audit implementation on the technical note's exact SAM counterexample (Sec. 2-3).

Lookup model f_w(e_j) = w_j, squared loss, memorized labels w_j=(1-delta_j)y_j with unequal residuals.  With plain SAM sharpness the ratio
s_t^2/L_t = 2 rho^2 / B for every probe step, so F == 0 exactly (up to fp error) although the model has no
generalization.  With ASAM sharpness the ratio is a residual-weighted average of w_i^2, so F != 0 in general.
"""
import torch

from common import AuditCfg, run_audit

torch.manual_seed(0)
N, M, delta = 400, 5000, 0.1
idx = torch.randperm(M)[:N]
X = torch.zeros(N, M)
X[torch.arange(N), idx] = 1.0
y = torch.randint(0, 2, (N,)).float() * 2 - 1
lin = torch.nn.Linear(M, 1, bias=False)
with torch.no_grad():
    lin.weight.zero_()
    lin.weight[0, idx] = (1 - delta * (0.5 + 4 * torch.rand(N))) * y   # unequal residuals / magnitudes
loss_fn = lambda out, t: 0.5 * ((out.squeeze(1) - t) ** 2).mean()
params = {k: v.detach().clone() for k, v in lin.named_parameters()}
for adaptive in (False, True):
    cfg = AuditCfg(adaptive=adaptive)
    Fs, h = [], None
    for s in range(5):
        f, h = run_audit(lin, params, X, y, s, cfg, loss_fn=loss_fn, record=True)
        Fs.append(float(f))
    print(f"adaptive(ASAM)={adaptive}:  F over 5 seeds = {[round(v, 6) for v in Fs]}")
    print(f"   ratios s^2/L (seed 4): {[round(r['ratio'], 6) for r in h]}   2*rho^2/B = {2 * cfg.rho ** 2 / cfg.B:.6f}")
