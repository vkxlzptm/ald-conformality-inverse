"""Explanatory figure: what the least-squares baseline actually does.

(a) the four simplex moves, schematically
(b) the real sum-of-squares landscape of this problem on a 2D slice, with the
    Nelder-Mead path that the optimiser actually walked
(c) how slowly the error falls per simulator call

The landscape is expensive (grid^2 simulator calls), so it is cached in
results/nm_fig_data.npz and reused on a re-run.

    python src/fig_nelder_mead.py            # default 22 x 22 grid
    python src/fig_nelder_mead.py --grid 8   # quick check
"""
import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import minimize

import baseline_ls as B
import generate_dataset as G
from cyl_run import run_ckpt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "results", "nm_fig_data.npz")
OUTPNG = os.path.join(ROOT, "docs", "nelder_mead_explained.png")

TRUTH = dict(AR=15.0, s0_ref=0.05, Ea=30.0, n_steric=2.0, reemit=0.97,
             n_sites=4.0)
DOSE = B.dose_of(float(G.PI2_CKPT[B.OBS_CKPT]), TRUTH["n_sites"],
                 TRUTH["AR"])


# --------------------------------------------------------------- computation
def compute(grid):
    run_ckpt(10., .05, 1., 1., 400., 1., np.array([50], np.int64), 50, .999, 1)

    clean = B.forward(TRUTH, TRUTH["AR"], DOSE, seed=555)
    orng = np.random.default_rng(9)
    obs = np.clip(clean * (1 + B.MEAS_NOISE * orng.standard_normal(clean.shape)),
                  0, 1)
    mask = np.ones(B.NBIN, bool)
    mask[orng.choice(B.NBIN, B.N_MASKED, replace=False)] = False

    s0r = np.geomspace(0.012, 0.20, grid)
    nsr = np.linspace(0.2, 4.2, grid)

    def sse2(ls0, nst):
        p = dict(TRUTH)
        p["s0_ref"] = float(np.exp(ls0))
        p["n_steric"] = float(nst)
        return B.sse(p, obs, mask, TRUTH["AR"], DOSE)

    t0 = time.time()
    Z = np.empty((grid, grid))
    for i, nst in enumerate(nsr):
        for j, s0 in enumerate(s0r):
            Z[i, j] = sse2(np.log(s0), nst)
        print(f"  landscape row {i+1}/{grid}  ({time.time()-t0:.0f} s)",
              flush=True)

    path, hist = [], []

    def obj2(v):
        f = sse2(v[0], v[1])
        path.append(v.copy())
        hist.append(f)
        return f

    x0 = np.array([np.log(0.16), 0.6])
    sim0 = np.vstack([x0, x0 + [0.9, 0.0], x0 + [0.0, 1.1]])
    minimize(obj2, x0, method="Nelder-Mead",
             options=dict(maxfev=140, xatol=1e-4, fatol=1e-8, adaptive=True,
                          initial_simplex=sim0))

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, Z=Z, s0r=s0r, nsr=nsr, path=np.array(path),
             hist=np.array(hist))
    print(f"cached -> {CACHE}  ({time.time()-t0:.0f} s)", flush=True)


# ------------------------------------------------------------------ drawing
def panel_simplex(ax):
    """Schematic of one Nelder-Mead iteration on a 2-parameter problem."""
    W = np.array([0.25, 0.18])          # worst corner
    B_ = np.array([0.18, 0.62])         # best
    G_ = np.array([0.64, 0.50])         # the other one
    M = (B_ + G_) / 2                   # centroid of all but the worst
    R = M + (M - W)                     # reflection
    E = M + 1.6 * (M - W)               # expansion
    C = M + 0.45 * (M - W)              # contraction

    ax.add_patch(Polygon([B_, G_, W], closed=True, fc="#d6e4f0",
                         ec="#2874a6", lw=2, zorder=2))
    ax.add_patch(Polygon([B_, B_ + 0.5 * (G_ - B_), B_ + 0.5 * (W - B_)],
                         closed=True, fc="none", ec="#c0392b", lw=1.6,
                         ls="--", zorder=3))
    ax.annotate("", xy=E, xytext=W,
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#2874a6",
                                shrinkA=6, shrinkB=2))

    for p_, lab, col, off in [
            (B_, "best", "#1e8449", (-0.02, 0.07)),
            (G_, "good", "#b9770e", (0.05, 0.04)),
            (W, "worst", "#c0392b", (-0.15, -0.03))]:
        ax.plot(*p_, "o", ms=12, color=col, zorder=6)
        ax.annotate(lab, p_ + np.array(off), ha="center", fontsize=11.5,
                    color=col, fontweight="bold")

    ax.plot(*M, "x", ms=11, mew=2.6, color="k", zorder=6)
    ax.annotate("centroid of" + "\n" + "the others", M + [0.16, -0.26],
                fontsize=9.5, ha="left")

    for p_, lab, col, off in [
            (C, "contract", "#8e44ad", (0.06, -0.03)),
            (R, "reflect", "#2874a6", (0.06, -0.01)),
            (E, "expand", "#117a65", (0.06, -0.01))]:
        ax.plot(*p_, "o", ms=10, mfc="white", mew=2.2, color=col, zorder=6)
        ax.annotate(lab, p_ + np.array(off), fontsize=10.5, color=col,
                    va="center")

    ax.annotate("shrink toward the best" + "\n" + "if none of those help",
                [-0.08, -0.13], fontsize=9.5, color="#c0392b", ha="left")

    ax.set_xlim(-0.10, 1.05)
    ax.set_ylim(-0.30, 1.24)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) One Nelder-Mead step" + "\n"
                 + "throw the worst corner across the others," + "\n"
                 + "then expand, contract or shrink", fontsize=12)


def draw():
    d = np.load(CACHE)
    Z, s0r, nsr, path, hist = (d["Z"], d["s0r"], d["nsr"], d["path"],
                               d["hist"])
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.6))
    fig.subplots_adjust(wspace=0.30)

    panel_simplex(axes[0])

    ax = axes[1]
    X, Y = np.meshgrid(np.log10(s0r), nsr)
    cf = ax.contourf(X, Y, np.log10(Z), levels=24, cmap="viridis")
    ax.contour(X, Y, np.log10(Z), levels=12, colors="w", linewidths=0.4,
               alpha=0.45)
    fig.colorbar(cf, ax=ax, label="Sum of squared residuals (log$_{10}$)")
    p = np.column_stack([path[:, 0] / np.log(10), path[:, 1]])
    ax.plot(p[:, 0], p[:, 1], "-", lw=1.1, color="w", alpha=0.85)
    ax.plot(p[:, 0], p[:, 1], "o", ms=3.2, color="#e74c3c")
    ax.plot(p[0, 0], p[0, 1], "s", ms=11, mfc="none", mew=2.4, color="#e74c3c")
    ax.annotate("start", p[0] + [0.02, 0.16], color="#e74c3c", fontsize=10.5,
                fontweight="bold")
    ax.plot(np.log10(TRUTH["s0_ref"]), TRUTH["n_steric"], "*", ms=20,
            color="#f4d03f", mec="k", mew=0.8, zorder=6)
    ax.annotate("truth", (np.log10(TRUTH["s0_ref"]), TRUTH["n_steric"]),
                xytext=(6, -18), textcoords="offset points", fontsize=10.5,
                color="w", fontweight="bold")
    ax.set_xlabel(r"Sticking probability  log$_{10}\,s_0$", fontsize=12)
    ax.set_ylabel("Steric exponent  n", fontsize=12)
    ax.set_title(f"(b) The actual landscape, and the path walked\n"
                 f"{len(path)} simulator calls on a 2D slice "
                 f"(the real fit is 5D)", fontsize=12)

    ax = axes[2]
    best = np.minimum.accumulate(hist)
    ax.plot(np.arange(1, len(hist) + 1), hist, lw=0.8, alpha=0.4,
            color="#7f8c8d", label="Each trial")
    ax.plot(np.arange(1, len(best) + 1), best, lw=2.4, color="#2874a6",
            label="Best so far")
    ax.set_yscale("log")
    ax.set_xlabel("Simulator calls", fontsize=12)
    ax.set_ylabel("Sum of squared residuals", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10.5)
    ax.set_title("(c) Progress per simulator call\n"
                 "no gradient is available, so every step costs a full "
                 "simulation", fontsize=12)

    fig.suptitle("What the least-squares baseline does — and why it is slow",
                 fontsize=15, y=1.04)
    os.makedirs(os.path.dirname(OUTPNG), exist_ok=True)
    fig.savefig(OUTPNG, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved ->", OUTPNG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=22)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    if a.redo or not os.path.exists(CACHE):
        compute(a.grid)
    else:
        print(f"using cached landscape {CACHE} (--redo to recompute)")
    draw()
