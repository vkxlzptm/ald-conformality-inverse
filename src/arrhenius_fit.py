"""Activation energy from a temperature split -- outside the network.

The network reads one wafer at a time and returns a posterior over the sticking
probability at that wafer's temperature.  The temperature dependence is then a
two-parameter straight line,

    ln s0 = ln A - (Ea / k_B) * (1 / T) ,

so it is fitted in closed form rather than learned.  Uncertainty is propagated by
refitting over draws from the per-wafer posteriors, which needs no simulator at
all.  Doing it this way means the result works for any temperatures and any
number of them, without retraining.

    python src/arrhenius_fit.py
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
from evaluate import load_model, posterior

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = np.array([0.1, 0.3, 0.5, 0.68, 0.8, 0.9, 0.95])


def fit_slopes(inv_T, ln_s0):
    """Least-squares slope of ln s0 against 1/T.

    inv_T : (G, K)      ln_s0 : (G, K, S)   ->  (G, S)
    """
    x = inv_T[:, :, None]
    xm = x.mean(1, keepdims=True)
    ym = ln_s0.mean(1, keepdims=True)
    num = ((x - xm) * (ln_s0 - ym)).sum(1)
    den = ((x - xm) ** 2).sum(1)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "results", "dataset"))
    ap.add_argument("--model", default=os.path.join(ROOT, "results", "model",
                                                    "best.pt"))
    ap.add_argument("--samples", type=int, default=400)
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
    samp, _ = posterior(net, x, c, dev, n_samp=a.samples)      # (N, S, 4)

    j = D.TARGETS.index("s0")
    z0 = samp[:, :, j] * zsd[j] + zmu[j]
    ln_s0 = D.ln_s0_from_z0(z0, d["AR"][:, None])  # s0 or Pi1 parametrisation

    # group the wafers that belong to one temperature split
    groups, inv = np.unique(d["group"], return_inverse=True)
    G, K, S = len(groups), D.K_TEMP, a.samples
    order = np.argsort(inv * 1e6 + d["T"])         # group-major, then ascending T
    gi = inv[order].reshape(G, K)
    assert (gi == gi[:, :1]).all(), "each group must hold one wafer per temperature"
    invT = (1.0 / d["T"][order]).reshape(G, K)
    lns = ln_s0[order].reshape(G, K, S)
    Ea_true = d["Ea_true"][order].reshape(G, K)[:, 0]

    Ea = -D.KB_EV * fit_slopes(invT, lns)          # (G, S), eV
    est = np.median(Ea, 1)
    lo68, hi68 = np.quantile(Ea, [0.16, 0.84], axis=1)
    Tbar = 1.0 / invT.mean(1)
    eta_true = Ea_true / (D.KB_EV * Tbar)
    eta_est = est / (D.KB_EV * Tbar)

    print(f"temperature splits: {G:,}   posterior draws per wafer: {S}")
    print(f"median |Ea - truth| : {np.median(np.abs(est - Ea_true))*1e3:.1f} meV"
          f"   ({np.median(np.abs(est - Ea_true))*D.KJMOL_PER_EV:.2f} kJ/mol)")
    print(f"median |eta - truth|: {np.median(np.abs(eta_est - eta_true)):.2f}"
          f"   (eta = Ea / k_B T, at the mean measurement temperature)")
    print(f"median 68 % interval width: {(np.median(hi68-lo68))*1e3:.1f} meV")

    print("\n  calibration of the Ea interval")
    cov = []
    for q in LEVELS:
        lo, hi = np.quantile(Ea, [0.5 - q / 2, 0.5 + q / 2], axis=1)
        f = float(((Ea_true >= lo) & (Ea_true <= hi)).mean())
        cov.append(f)
        print(f"   stated {q*100:4.0f} %   ->  contains truth {f*100:5.1f} %")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))

    ax = axes[0]
    ax.errorbar(Ea_true * 1e3, est * 1e3,
                yerr=[(est - lo68) * 1e3, (hi68 - est) * 1e3],
                fmt="o", ms=3, lw=0.6, alpha=0.30, color="#2874a6")
    lim = [0, max(Ea_true.max(), est.max()) * 1.05 * 1e3]
    ax.plot(lim, lim, "k--", lw=1.3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True activation energy  (meV)", fontsize=12)
    ax.set_ylabel("Inferred, with 68 % interval  (meV)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_title("Activation energy from three wafers\n"
                 "fitted in closed form, never learned", fontsize=12)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1.3, label="Perfect calibration")
    ax.plot(LEVELS, cov, "o-", ms=6, lw=2, color="#c0392b", label="$E_a$")
    ax.set_xlabel("Stated central interval", fontsize=12)
    ax.set_ylabel("Fraction containing the truth", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=10)
    ax.set_title("Uncertainty propagated from the wafer posteriors\n"
                 "no simulator call anywhere in this step", fontsize=12)

    ax = axes[2]
    g = int(np.argsort(np.abs(Ea_true - np.median(Ea_true)))[0])
    xx = invT[g] * 1e3
    for s in range(min(120, S)):
        ax.plot(xx, lns[g, :, s] / np.log(10), "-", lw=0.6, alpha=0.07,
                color="#2874a6")
    ax.plot(xx, np.median(lns[g], axis=1) / np.log(10), "o-", ms=7, lw=2,
            color="#154360", label="Posterior median")
    ax.set_xlabel(r"1000 / T   (K$^{-1}$)", fontsize=12)
    ax.set_ylabel(r"log$_{10}\ s_0$", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=10)
    ax.set_title(f"One split, {min(120, S)} posterior draws\n"
                 f"true $E_a$ = {Ea_true[g]*1e3:.0f} meV, "
                 f"inferred {est[g]*1e3:.0f} "
                 f"[{lo68[g]*1e3:.0f}, {hi68[g]*1e3:.0f}]", fontsize=12)

    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "arrhenius.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
