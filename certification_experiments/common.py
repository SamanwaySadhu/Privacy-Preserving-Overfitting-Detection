"""Shared code for the directional-sharpness *certification* experiments.

Certification scenario (paper Sec. 3.1 / 5.3): the verifier never sees the
training set D.  The prover holds (w0, D), runs the public audit (Algorithm 1)
with a public seed r and proves  F(w0, D, r) <= tau.  Everything the verifier
learns is that single predicate, so we model the verifier as: pick tau on a
calibration set of models of known quality, then accept iff F <= tau.

The audit below is a from-scratch, *differentiable* re-implementation of
Algorithm 1 with the paper's defaults (B=16, T=10, lr=1e-4, rho=0.05, ASAM
sharpness, fluc = Std(log(r_t+eps)), r_t = s_t^2 / L_t).  It is needed so the
adaptive attack can back-propagate through the probe.  `cross_check.py`
compares it against d-sharpness-code/df_utils/df.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
REPO = ROOT.parent


# ----------------------------------------------------------------- models
class LeNet5(nn.Module):  # as in image_model_experiments/spectral_norm_lenet5_cifar10.ipynb
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class SimpleCNN(nn.Module):  # as in image_model_experiments/spectral_norm_1cnn_fmnist.ipynb
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        return self.fc1(x.flatten(1))


class TwoFC(nn.Module):  # TabularNet (128, 64) from tabular_model_experiments
    def __init__(self, d_in, n_out=2):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                   nn.Linear(64, n_out))

    def forward(self, x):
        return self.model(x)


class _Block(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.conv1 = nn.Conv2d(i, o, 3, s, 1)
        self.bn1 = nn.BatchNorm2d(o)
        self.conv2 = nn.Conv2d(o, o, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(o)
        self.shortcut = nn.Sequential()
        if s != 1 or i != o:
            self.shortcut = nn.Sequential(nn.Conv2d(i, o, 1, s), nn.BatchNorm2d(o))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """The repo's 'ResNet18' = Cifar10NarrowResNet(k=1): 3 stages x 3 basic blocks, widths 16/32/64."""

    def __init__(self, k=1, n_out=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16 * k, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16 * k)
        layers, inp = [], 16 * k
        for planes, stride in ((16 * k, 1), (32 * k, 2), (64 * k, 2)):
            for j in range(3):
                layers.append(_Block(inp, planes, stride if j == 0 else 1))
                inp = planes
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(inp, n_out)

    def forward(self, x):
        x = self.layers(F.relu(self.bn1(self.conv1(x))))
        return self.linear(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ------------------------------------------------------------------- data
@dataclass
class Task:
    name: str
    arch: str
    dataset: str
    n_train: int
    noise: float          # label-noise rate of the faulty (memorizing) model
    lr: float
    epochs: int           # max epochs
    target_train_acc: float
    n_q: int = 512        # attacker's auxiliary set Q (disjoint from D and the test set)


TASKS = {
    "lenet5":    Task("lenet5", "lenet5", "cifar10", 4000, 0.2, 0.02, 150, 0.90),
    "resnet18":  Task("resnet18", "resnet18", "cifar10", 4000, 0.2, 0.05, 60, 0.90),
    "simplecnn": Task("simplecnn", "simplecnn", "fmnist", 4000, 0.2, 0.02, 150, 0.90),
    "2fc":       Task("2fc", "2fc", "german", 640, 0.2, 0.02, 400, 0.90, n_q=160),
}


def build_model(arch, d_in=None):
    if arch == "2fc":
        return TwoFC(d_in)
    return {"lenet5": LeNet5, "simplecnn": SimpleCNN, "resnet18": ResNet18}[arch]()


def _load_torchvision(dataset):
    import torchvision
    from torchvision import transforms as T
    from PIL import Image
    if dataset == "cifar10":
        tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        cls = torchvision.datasets.CIFAR10
        arr = lambda a: a
    else:
        tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        cls = torchvision.datasets.FashionMNIST
        arr = lambda a: a.numpy()
    out = []
    for train in (True, False):
        d = cls(str(DATA), train, download=True)
        X = torch.stack([tf(Image.fromarray(arr(a))) for a in d.data])
        out.append((X, torch.as_tensor(d.targets)))
    return out


def _load_german():
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    cols = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "EmploymentSince",
            "InstallmentRate", "SexAndStatus", "OtherDebtors", "ResidenceSince", "Property", "Age",
            "OtherInstallmentPlans", "Housing", "ExistingCredits", "Job", "NumPeopleLiable", "Telephone",
            "ForeignWorker", "Target"]
    df = pd.read_csv(REPO / "german.data", sep=r"\s+", header=None, names=cols)
    y = (df.pop("Target") - 1).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)
    cat = Xtr.select_dtypes(include="object").columns
    num = Xtr.select_dtypes(exclude="object").columns
    pp = ColumnTransformer([("n", StandardScaler(), num),
                            ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    Xtr = torch.tensor(pp.fit_transform(Xtr), dtype=torch.float32)
    Xte = torch.tensor(pp.transform(Xte), dtype=torch.float32)
    return (Xtr, torch.tensor(ytr.values)), (Xte, torch.tensor(yte.values))


def load_task_data(task: Task):
    """D (clean labels), D_noisy (memorization labels), test split, and the attacker's aux set Q."""
    (Xtr, ytr), (Xte, yte) = _load_german() if task.dataset == "german" else _load_torchvision(task.dataset)
    g = torch.Generator().manual_seed(1234)
    perm = torch.randperm(len(Xtr), generator=g)
    iD, iQ = perm[:task.n_train], perm[task.n_train:task.n_train + task.n_q]
    X, y = Xtr[iD], ytr[iD]
    n_cls = int(ytr.max()) + 1
    flip = torch.randperm(len(y), generator=g)[: int(task.noise * len(y))]
    y_noisy = y.clone()
    y_noisy[flip] = (y[flip] + torch.randint(1, n_cls, (len(flip),), generator=g)) % n_cls
    return dict(X=X, y=y, y_noisy=y_noisy, Xte=Xte, yte=yte, Q=Xtr[iQ], n_cls=n_cls)


# ------------------------------------------------------------ train / eval
@torch.no_grad()
def evaluate(model, X, y, bs=1000):
    model.eval()
    tl, tc = 0.0, 0
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs])
        tl += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        tc += (out.argmax(1) == y[i:i + bs]).sum().item()
    return tl / len(X), tc / len(X)


def train_model(task: Task, data, labels, seed, log=print):
    """Plain SGD (mom .9, wd 5e-4) until train-acc (on `labels`) >= target, as in paper Sec. 5.1."""
    torch.manual_seed(seed)
    model = build_model(task.arch, data["X"].shape[1] if task.arch == "2fc" else None)
    opt = torch.optim.SGD(model.parameters(), lr=task.lr, momentum=0.9, weight_decay=5e-4)
    X = data["X"]
    n = len(X)
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, task.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            F.cross_entropy(model(X[idx]), labels[idx]).backward()
            opt.step()
        tl, ta = evaluate(model, X, labels)
        if ta >= task.target_train_acc:
            break
    log(f"    trained seed={seed} epochs={ep} train_loss={tl:.3f} train_acc={ta:.3f}")
    model.eval()
    return model, ep


def model_report(model, data, labels):
    trl, tra = evaluate(model, data["X"], labels)
    tel, tea = evaluate(model, data["Xte"], data["yte"])
    _, tra_true = evaluate(model, data["X"], data["y"])  # vs. the *true* train labels
    return dict(train_loss=trl, train_acc=tra, train_acc_true_labels=tra_true, test_loss=tel, test_acc=tea,
                loss_gap=tel - trl, acc_gap=tra_true - tea)


# ------------------------------------------------------------------ audit
@dataclass
class AuditCfg:
    B: int = 16
    T: int = 10
    lr: float = 1e-4
    rho: float = 0.05
    momentum: float = 0.9
    adaptive: bool = True      # ASAM sharpness (paper default)
    eps: float = 1e-10         # epsilon in log(r_t + eps)
    stat: str = "log"          # 'log'  : Std_t log(s_t^2/L_t + eps)   (paper text, Sec. 4.2)
                               # 'ratio': Std_t (s_t^2/L_t)            (artifact df.py: FLAT2_OVER_LOSS_STD)


def _loss(model, params, buffers, x, y, loss_fn):
    return loss_fn(functional_call(model, (params, buffers), (x,)), y)


def run_audit(model, params, X, y, seed, cfg: AuditCfg = AuditCfg(), differentiable=False,
              loss_fn=F.cross_entropy, record=False):
    """Algorithm 1 (BN frozen / eval mode).  `params`: dict name->tensor (graph-attached if differentiable).
    Batches: a public permutation of D seeded by `seed`, consecutive blocks of size B (the paper's ZK setting).
    Returns F = Std_t log(s_t^2/L_t + eps) (+ per-step history if record)."""
    buffers = dict(model.named_buffers())
    names = list(params.keys())
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(X), generator=g)
    assert cfg.T * cfg.B <= len(X)
    w = {k: (v if differentiable else v.detach().clone().requires_grad_(True)) for k, v in params.items()}
    mom = None
    us, rs, hist = [], [], []
    for t in range(cfg.T):
        idx = perm[t * cfg.B:(t + 1) * cfg.B]
        x, yb = X[idx], y[idx]
        loss = _loss(model, w, buffers, x, yb, loss_fn)
        grads = torch.autograd.grad(loss, [w[k] for k in names], create_graph=differentiable)
        tg = [w[k].abs() * gr for k, gr in zip(names, grads)] if cfg.adaptive else list(grads)
        sq = sum((a * a).sum() for a in tg)
        s2 = cfg.rho ** 2 * sq                                  # s_t^2
        r = s2 / loss.clamp_min(1e-12)
        us.append(torch.log(r + cfg.eps))
        rs.append(r)
        if record:
            hist.append(dict(loss=float(loss.detach()), s2=float(s2.detach()), ratio=float(r.detach())))
        nrm = torch.sqrt(sq) + 1e-12
        # SAM ascent: e = rho * T_w^2 g / ||T_w g|| (adaptive) or rho * g / ||g||
        wa = {k: w[k] + (w[k] ** 2 if cfg.adaptive else 1.0) * gr * (cfg.rho / nrm)
              for k, gr in zip(names, grads)}
        if not differentiable:
            wa = {k: v.detach().requires_grad_(True) for k, v in wa.items()}
        loss2 = _loss(model, wa, buffers, x, yb, loss_fn)
        g2 = torch.autograd.grad(loss2, [wa[k] for k in names], create_graph=differentiable)
        mom = list(g2) if mom is None else [cfg.momentum * m + gr for m, gr in zip(mom, g2)]
        if differentiable:
            w = {k: w[k] - cfg.lr * m for k, m in zip(names, mom)}
        else:
            w = {k: (w[k].detach() - cfg.lr * m.detach()).requires_grad_(True) for k, m in zip(names, mom)}
    Fval = torch.stack(us if cfg.stat == "log" else rs).std(unbiased=False)
    return (Fval, hist) if record else Fval


def model_params(model):
    return {k: v.detach().clone() for k, v in model.named_parameters()}


def F_over_seeds(model, X, y, seeds, cfg=AuditCfg()):
    p = model_params(model)
    model.eval()
    return np.array([float(run_audit(model, p, X, y, s, cfg)) for s in seeds])


def F_both(model, X, y, seeds, cfg=AuditCfg()):
    """Both statistics from one probe trajectory per seed -> {'log': array, 'ratio': array}."""
    p = model_params(model)
    model.eval()
    out = {"log": [], "ratio": []}
    for sd in seeds:
        _, h = run_audit(model, p, X, y, sd, cfg, record=True)
        r = np.array([x["ratio"] for x in h])
        out["log"].append(float(np.std(np.log(r + cfg.eps))))
        out["ratio"].append(float(np.std(r)))
    return {k: np.array(v) for k, v in out.items()}


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
