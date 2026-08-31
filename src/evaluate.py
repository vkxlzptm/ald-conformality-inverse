"""Evaluate the trained network: accuracy, calibration, and a head-to-head
comparison with the least-squares baseline on the same benchmark cases.

    python src/evaluate.py                       # after train.py
    python src/evaluate.py --cases 8             # also runs the benchmark set

Calibration is the part that matters: claiming "uncertainty quantification"
means nothing unless the stated intervals actually contain the truth at the
stated rate.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
from model import ProfileMDN, mdn_nll, mdn_sample

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
PRETTY = {"s0": "$s_0$", "n_steric": "n", "reemit": "re-emission",
          "pi2": r"$\Pi_2$ (dose per site)"}


def load_model(path, dev):
    """Load a checkpoint and put data.py into the parametrisation it was trained in.

    Doing it here means evaluate, the Arrhenius fit and the transfer study all
    pick it up from the file rather than from a flag someone has to remember.
    """
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    D.set_target_param(a.get("target_param", "s0"))
    net = ProfileMDN(n_out=len(ck["targets"]), n_mix=a["mix"],
                     width=a["width"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, np.asarray(ck["zmu"]), np.asarray(ck["zsd"])


@torch.no_grad()
def posterior(net, x, c, dev, n_samp=400, bs=512):
    """Posterior samples in z space: (N, n_samp, 5), plus NLL inputs."""
    out, heads = [], []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(dev)
        cb = torch.from_numpy(c[i:i + bs]).to(dev)
        h = net(xb, cb)
        out.append(mdn_sample(*h, n_samp).cpu().numpy())
        heads.append([t.cpu() for t in h])
    return np.concatenate(out), heads


def coverage(samples_z, truth_z):
    """Fraction of truths inside the central interval, per level and parameter."""
    cov = np.zeros((len(LEVELS), truth_z.shape[1]))
    for i, q in enumerate(LEVELS):
        lo = np.quantile(samples_z, 0.5 - q / 2, axis=1)
        hi = np.quantile(samples_z, 0.5 + q / 2, axis=1)
        cov[i] = ((truth_z >= lo) & (truth_z <= hi)).mean(0)
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "results", "dataset"))
    ap.add_argument("--model", default=os.path.join(ROOT, "results", "model",
                                                    "best.pt"))
    ap.add_argument("--cases", type=int, default=0,
                    help="also run the least-squares benchmark cases")
    ap.add_argument("--baseline-json", default=os.path.join(ROOT, "results",
                                                            "baseline.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device(a.device)
    net, zmu, zsd = load_model(a.model, dev)
    _, _, te_ids = D.split_ids(a.data)
    d = D.load(a.data, te_ids)
    rng = np.random.default_rng(a.seed)
    obs, mask = D.degrade(d["y"], d["mc_noise"], rng)
    x, c = D.features(obs, mask, d["AR"])
    z_true = (D.targets_to_z(d["p"], d["AR"]) - zmu) / zsd

    t0 = time.time()
    samp, heads = posterior(net, x, c, dev)
    t_inf = (time.time() - t0) / len(x)

    nll = float(np.mean([
        mdn_nll(*h, torch.from_numpy(
            z_true[i * 512:i * 512 + h[0].shape[0]].astype(np.float32))).item()
        for i, h in enumerate(heads)]))

    mean_z = samp.mean(1)
    est = D.z_to_targets(mean_z * zsd + zmu, d["AR"])
    truth = d["p"]
    print(f"test set: {len(x):,} examples from {len(te_ids)} shards")
    print(f"first target trained as: {D.TARGET_PARAM}  "
          f"(errors below are always in physical units, so they compare across "
          f"parametrisations; the mixture NLL does not)")
    print(f"inference: {t_inf*1e3:.3f} ms per measurement on {dev}")
    print(f"mixture NLL (normalized space): {nll:.4f}\n")
    print("  targets are dimensionless; the site density follows from")
    print("  n_s = N_dose / (4 AR Pi2) with the measured dose.\n")
    print("  parameter        median error   posterior width (68 %)")
    lo = np.quantile(samp, 0.16, axis=1)
    hi = np.quantile(samp, 0.84, axis=1)
    for j, name in enumerate(D.TARGETS):
        if name == "reemit":                 # sits near 1; relative error misleads
            err = np.abs(est[:, j] - truth[:, j])
            unit, val = "abs", np.median(err)
        else:
            err = np.abs(est[:, j] - truth[:, j]) / np.maximum(truth[:, j], 1e-9)
            unit, val = "%", np.median(err) * 100
        w = np.median(hi[:, j] - lo[:, j]) * zsd[j]
        print(f"  {name:11s}   {val:9.1f} {unit:7s}   {w:.3f}  (z units)")

    cov = coverage(samp, z_true)
    print("\n  calibration -- fraction of truths inside the central interval")
    print("   level  " + "  ".join(f"{n:>9s}" for n in D.TARGETS))
    for i, q in enumerate(LEVELS):
        print(f"   {q*100:4.0f}%  " +
              "  ".join(f"{cov[i, j]*100:8.1f}%" for j in range(len(D.TARGETS))))

    # ------------------------------------------------- benchmark comparison
    bench = None
    if a.cases:
        import baseline_ls as B
        xs, cs, tr, ARs = [], [], [], []
        for i in range(a.cases):
            t, o, m, AR, _T = B.make_case(i)
            xi, ci = D.features(o[None].astype(np.float32),
                                m[None].astype(np.float32), np.array([AR]))
            xs.append(xi); cs.append(ci); ARs.append(AR)
            tr.append([t[k] for k in D.TARGETS])
        xb = np.concatenate(xs); cb = np.concatenate(cs)
        tr = np.asarray(tr); ARs = np.asarray(ARs)
        t0 = time.time()
        sb, _ = posterior(net, xb, cb, dev)
        t_case = (time.time() - t0) / len(xb)
        eb = D.z_to_targets(sb.mean(1) * zsd + zmu, ARs)
        bench = dict(truth=tr, net=eb, t=t_case)
        print(f"\n  benchmark cases: {a.cases}, "
              f"{t_case*1e3:.2f} ms each by the network")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2 if bench is None else 3,
                             figsize=(6.2 * (2 if bench is None else 3), 5.2))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.3, label="Perfect calibration")
    for j, name in enumerate(D.TARGETS):
        ax.plot(LEVELS, cov[:, j], "o-", ms=5, lw=1.8, label=PRETTY[name])
    ax.set_xlabel("Stated central interval", fontsize=12)
    ax.set_ylabel("Fraction of truths inside it", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title("Calibration on the held-out shards\n"
                 "above the line is under-confident, below is over-confident",
                 fontsize=12)

    ax = axes[1]
    j = D.TARGETS.index("s0")
    ax.errorbar(truth[:, j], est[:, j],
                yerr=[est[:, j] - D.z_to_targets(lo * zsd + zmu, d["AR"])[:, j],
                      D.z_to_targets(hi * zsd + zmu, d["AR"])[:, j] - est[:, j]],
                fmt="o", ms=3, lw=0.7, alpha=0.35, color="#2874a6")
    lim = [truth[:, j].min() * 0.7, truth[:, j].max() * 1.4]
    ax.plot(lim, lim, "k--", lw=1.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("True $s_0$", fontsize=12)
    ax.set_ylabel("Inferred $s_0$, with 68 % interval", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Recovery of the sticking probability", fontsize=12)

    if bench is not None:
        ax = axes[2]
        base = None
        if os.path.exists(a.baseline_json):
            with open(a.baseline_json) as fh:
                base = json.load(fh)
        w = 0.38
        idx = np.arange(len(D.TARGETS))
        netv = []
        for j, name in enumerate(D.TARGETS):
            e = np.abs(bench["net"][:, j] - bench["truth"][:, j])
            netv.append(np.median(e * 100 if name == "reemit"
                                  else e / np.maximum(bench["truth"][:, j], 1e-9)
                                  * 100))
        ax.bar(idx - w / 2, netv, w, label="Amortized network", color="#2874a6")
        if base:
            bv = []
            for name in D.TARGETS:
                e = [abs(cc["est"][name] - cc["truth"][name]) * 100
                     for cc in base["cases"]]
                r = [abs(cc["est"][name] - cc["truth"][name]) /
                     max(abs(cc["truth"][name]), 1e-9) * 100
                     for cc in base["cases"]]
                bv.append(np.median(e if name == "reemit" else r))
            ax.bar(idx + w / 2, bv, w, label="Least squares", color="#c0392b")
        ax.set_xticks(idx)
        ax.set_xticklabels([PRETTY[n] for n in D.TARGETS], fontsize=9)
        ax.set_ylabel("Median error  (%; re-emission in points)", fontsize=12)
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=10)
        ax.set_title("Same benchmark cases, head to head", fontsize=12)

    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "evaluation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
