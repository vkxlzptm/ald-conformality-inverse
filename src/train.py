"""Train the amortized inference network.

    python src/train.py --data results/dataset --epochs 60
    python src/train.py --device cpu --epochs 2 --limit-shards 30   # timing probe

Splits by shard (never by example -- see src/data.py), applies the measurement
model fresh every epoch as augmentation, and reports seconds per epoch so the
CPU/GPU choice can be made on measurement rather than on assumption.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import data as D
from model import ProfileMDN, mdn_nll, mdn_moments

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Split:
    def __init__(self, root, ids, zmu=None, zsd=None):
        d = D.load(root, ids)
        self.y, self.AR, self.dose = d["y"], d["AR"], d["dose"]
        self.mc, self.shard = d["mc_noise"], d["shard"]
        self.p = d["p"]
        z = D.targets_to_z(self.p)
        self.zmu = z.mean(0) if zmu is None else zmu
        # guard: a split with a single parameter draw has zero spread
        self.zsd = np.maximum(z.std(0), 1e-8) if zsd is None else zsd
        self.z = ((z - self.zmu) / self.zsd).astype(np.float32)
        self.n = len(self.y)

    def batches(self, bs, rng, shuffle=True):
        obs, mask = D.degrade(self.y, self.mc, rng)
        x, c = D.features(obs, mask, self.AR, self.dose)
        order = rng.permutation(self.n) if shuffle else np.arange(self.n)
        for i in range(0, self.n, bs):
            j = order[i:i + bs]
            yield x[j], c[j], self.z[j]


def run_epoch(net, sp, bs, rng, dev, opt=None, sched=None):
    train = opt is not None
    net.train(train)
    tot, nb = 0.0, 0
    for xb, cb, zb in sp.batches(bs, rng, shuffle=train):
        xb = torch.from_numpy(xb).to(dev)
        cb = torch.from_numpy(cb).to(dev)
        zb = torch.from_numpy(zb).to(dev)
        with torch.set_grad_enabled(train):
            loss = mdn_nll(*net(xb, cb), zb)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if sched is not None:
                sched.step()
        tot += loss.item(); nb += 1
    return tot / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "results", "dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "model"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--mix", type=int, default=8)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="use only the first N shards (quick timing probe)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = pick_device(a.device)
    tr_ids, va_ids, te_ids = D.split_ids(a.data)
    if a.limit_shards:
        keep = set(sorted(tr_ids + va_ids + te_ids)[:a.limit_shards])
        tr_ids = [i for i in tr_ids if i in keep]
        va_ids = [i for i in va_ids if i in keep] or tr_ids[-1:]
        tr_ids = [i for i in tr_ids if i not in set(va_ids)]

    t0 = time.time()
    tr = Split(a.data, tr_ids)
    va = Split(a.data, va_ids, tr.zmu, tr.zsd)
    print(f"device {dev}   train {tr.n:,} ex / {len(tr_ids)} shards   "
          f"val {va.n:,} ex / {len(va_ids)} shards   "
          f"test {len(te_ids)} shards   (load {time.time()-t0:.1f} s)",
          flush=True)

    net = ProfileMDN(n_mix=a.mix, width=a.width).to(dev)
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}",
          flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(tr.n / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    best, hist = np.inf, []
    for ep in range(1, a.epochs + 1):
        t = time.time()
        trl = run_epoch(net, tr, a.batch, rng, dev, opt, sched)
        val = run_epoch(net, va, a.batch, rng, dev)
        dt = time.time() - t
        hist.append(dict(epoch=ep, train=trl, val=val, sec=dt))
        tag = ""
        if val < best:
            best = val
            torch.save(dict(state=net.state_dict(), zmu=tr.zmu, zsd=tr.zsd,
                            args=vars(a), targets=D.TARGETS),
                       os.path.join(a.out, "best.pt"))
            tag = "  *"
        print(f"  epoch {ep:3d}   train {trl:8.4f}   val {val:8.4f}   "
              f"{dt:5.1f} s{tag}", flush=True)

    with open(os.path.join(a.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    med = np.median([h["sec"] for h in hist])
    print(f"\nbest val NLL {best:.4f}   median {med:.1f} s/epoch on {dev}")


if __name__ == "__main__":
    main()
