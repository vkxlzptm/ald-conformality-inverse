"""Does expressing the first target as Pi1 = AR sqrt(s0) beat expressing it as s0?

AR is a network input either way, so the two parametrisations are in bijection
and a flexible network can build one from the other: this is not a question about
what the model *can* represent.  What actually changes is the quantity the loss
is z-normalised in.  Measured on the prior used here, log s0 has a spread of 1.51
and log Pi1 a spread of 0.88, so the two put the error budget in different
places -- and Pi1 is the group the profile shape was measured to collapse on
(README 4-2).  Whether that helps is an empirical question, so it is measured
rather than argued.

The comparison is paired and like for like:

  * the same held-out shards, and the *same* measurement noise and mask draw, so
    both models see byte-identical observations
  * errors reported in physical units (s0, n, re-emission, Pi2) for both, which
    is the only space in which the two are comparable
  * calibration compared as coverage of the truth, which is invariant under the
    monotone change of variable between the two parametrisations

The mixture NLL is not compared: each model is a density over its own z-scored
space, and the map between those spaces mixes in AR, so the two numbers are not
on the same scale.  Physical-unit errors and coverage are.

    python src/train.py --target-param pi1 --out results/model_pi1
    python src/ablation_param.py --models results/model/best.pt \\
                                          results/model_pi1/best.pt
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
from evaluate import load_model, posterior, LEVELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLORS = ["#2874a6", "#c0392b", "#1e8449", "#8e44ad"]
ABS_ERROR = {"reemit"}


def med_err(est, truth):
    out = []
    for j, name in enumerate(D.TARGETS):
        e = np.abs(est[:, j] - truth[:, j])
        out.append(float(np.median(e)) if name in ABS_ERROR
                   else float(np.median(e / np.maximum(truth[:, j], 1e-12)) * 100))
    return np.array(out)


def coverage(samples, truth):
    """Fraction of truths inside the central interval, per level and target.

    Computed on physical samples; coverage is invariant under the monotone map
    from either training space, so this is directly comparable.
    """
    cov = np.zeros((len(LEVELS), truth.shape[1]))
    for i, q in enumerate(LEVELS):
        lo = np.quantile(samples, 0.5 - q / 2, axis=1)
        hi = np.quantile(samples, 0.5 + q / 2, axis=1)
        cov[i] = ((truth >= lo) & (truth <= hi)).mean(0)
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "results", "dataset"))
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ar-bins", type=int, default=4)
    ap.add_argument("--json", default=os.path.join(ROOT, "results",
                                                   "ablation_param.json"))
    a = ap.parse_args()

    # Fail before any work: loading the test set and running the first model's
    # posterior takes minutes, and losing that to a typo in the second path is
    # pure waste.
    missing = [m for m in a.models if not os.path.exists(m)]
    if missing:
        raise SystemExit("no such checkpoint:\n  " + "\n  ".join(missing) +
                         "\n\ntrain it first, e.g.\n"
                         "  python src/train.py --target-param pi1 "
                         "--out results/model_pi1 --epochs <same as the control>")
    if len(a.models) != len(set(map(os.path.abspath, a.models))):
        raise SystemExit("the same checkpoint was passed twice")

    dev = torch.device(a.device)
    _, _, te_ids = D.split_ids(a.data)
    d = D.load(a.data, te_ids)

    # one observation draw, shared by every model: a paired comparison
    rng = np.random.default_rng(a.seed)
    obs, mask = D.degrade(d["y"], d["mc_noise"], rng)
    x, c = D.features(obs, mask, d["AR"])
    truth, AR = d["p"], d["AR"]
    print(f"test set : {len(x):,} wafers from {len(te_ids)} shards, "
          f"identical observations for every model")

    edges = np.quantile(np.log(AR), np.linspace(0, 1, a.ar_bins + 1))
    edges[-1] += 1e-9
    abin = np.clip(np.searchsorted(edges, np.log(AR), "right") - 1,
                   0, a.ar_bins - 1)

    runs = []
    for path, col in zip(a.models, COLORS):
        net, zmu, zsd = load_model(path, dev)      # sets D.TARGET_PARAM
        mode = D.TARGET_PARAM
        samp, _ = posterior(net, x, c, dev, n_samp=a.samples)     # (N, S, 4)
        n, S, k = samp.shape
        phys = D.z_to_targets((samp * zsd + zmu).reshape(-1, k),
                              np.repeat(AR, S)).reshape(n, S, k)
        est = phys.mean(1)
        js = D.TARGETS.index("s0")
        per_ar = [float(np.median(np.abs(est[abin == b, js] - truth[abin == b, js])
                                  / truth[abin == b, js]) * 100)
                  for b in range(a.ar_bins)]
        runs.append(dict(path=path, mode=mode, color=col,
                         err=med_err(est, truth), cov=coverage(phys, truth),
                         per_ar=per_ar))
        print(f"  loaded {os.path.relpath(path, ROOT)}  ->  first target: {mode}")

    # ------------------------------------------------------------------ table
    print("\n  median error, physical units "
          "(%, re-emission in absolute units)")
    print("   first target " + "  ".join(f"{n:>11s}" for n in D.TARGETS))
    for r in runs:
        print(f"   {r['mode']:11s} " +
              "  ".join(f"{v:11.2f}" for v in r["err"]))
    if len(runs) == 2:
        rel = (runs[1]["err"] / np.maximum(runs[0]["err"], 1e-12) - 1) * 100
        print(f"   {'change':11s} " +
              "  ".join(f"{v:+10.1f}%" for v in rel) +
              f"   ({runs[1]['mode']} relative to {runs[0]['mode']}, "
              f"negative is better)")

    print("\n  worst calibration deviation over the stated levels "
          "(0 = perfect)")
    print("   first target " + "  ".join(f"{n:>11s}" for n in D.TARGETS))
    for r in runs:
        dev_max = np.abs(r["cov"] - LEVELS[:, None]).max(0)
        print(f"   {r['mode']:11s} " + "  ".join(f"{v:11.3f}" for v in dev_max))

    lo = np.exp(edges[:-1]); hi = np.exp(edges[1:])
    print("\n  median s0 error by aspect ratio (%)")
    print("   first target " +
          "  ".join(f"{l:.0f}-{h:.0f}".rjust(11) for l, h in zip(lo, hi)))
    for r in runs:
        print(f"   {r['mode']:11s} " +
              "  ".join(f"{v:11.2f}" for v in r["per_ar"]))

    with open(a.json, "w") as fh:
        json.dump([dict(path=r["path"], mode=r["mode"],
                        err=list(map(float, r["err"])),
                        per_ar=r["per_ar"],
                        cov=r["cov"].tolist()) for r in runs], fh, indent=1)
    print(f"\n  wrote {a.json}")

    # ----------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    idx = np.arange(len(D.TARGETS))
    w = 0.8 / len(runs)
    ax = axes[0]
    for i, r in enumerate(runs):
        ax.bar(idx + (i - (len(runs) - 1) / 2) * w, r["err"], w,
               label=f"First target: {r['mode']}", color=r["color"])
    ax.set_xticks(idx)
    ax.set_xticklabels(["$s_0$", "n", "re-emission", r"$\Pi_2$"], fontsize=10)
    ax.set_ylabel("Median error  (%; re-emission in points)", fontsize=12)
    ax.set_title("Same observations, same split, physical units", fontsize=12)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1.3, label="Perfect calibration")
    js = D.TARGETS.index("s0")
    for r in runs:
        ax.plot(LEVELS, r["cov"][:, js], "o-", ms=5, lw=1.8, color=r["color"],
                label=f"{r['mode']}")
    ax.set_xlabel("Stated central interval", fontsize=12)
    ax.set_ylabel("Fraction of truths inside it", fontsize=12)
    ax.set_title("Calibration of $s_0$, comparable across\n"
                 "parametrisations because coverage is", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[2]
    ctr = np.sqrt(lo * hi)
    for r in runs:
        ax.plot(ctr, r["per_ar"], "o-", ms=6, lw=2, color=r["color"],
                label=f"{r['mode']}")
    ax.set_xscale("log")
    ax.set_xlabel("Aspect ratio", fontsize=12)
    ax.set_ylabel("Median error in $s_0$  (%)", fontsize=12)
    ax.set_title(r"Is the error budget more even in AR when the"
                 "\n" r"target is the group the shape collapses on?", fontsize=12)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=10)

    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "ablation_param.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
