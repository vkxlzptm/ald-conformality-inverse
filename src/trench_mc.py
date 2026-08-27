"""
2D trench ballistic Monte Carlo for ALD step coverage.

Knudsen (molecular flow) regime: no gas-gas collisions, so absolute dimensions
are irrelevant and only AR = H/W matters.  W is fixed to 1 and H = AR.

Notation follows Cremers, Puurunen & Dendooven, Appl. Phys. Rev. 6, 021302 (2019):
    s   = sticking probability (probability of reaction per wall collision)
    s0  = initial sticking probability on the bare surface
    s(theta) = s0 * (1 - theta)**n_steric
"""
import numpy as np
from numba import njit

NBIN = 64


@njit(cache=True)
def _launch_dir(u):
    """2D cosine law about the inward normal: p(a) ~ cos(a), a in [-pi/2, pi/2]."""
    return np.arcsin(2.0 * u - 1.0)


@njit(cache=True)
def run_mc(AR, n_part, s0, n_steric, reemit, sites_per_bin, weight,
           trace_n, seed):
    """
    Track n_part pseudo-particles, each carrying `weight` molecules.

    Returns
    -------
    theta : (NBIN,) coverage of the sidewall bins, 0..1
    tx, tz, tn : trajectory vertices of the first `trace_n` particles
    """
    np.random.seed(seed)
    H, W = AR, 1.0
    theta = np.zeros(NBIN)
    tx = np.zeros((trace_n, 200))
    tz = np.zeros((trace_n, 200))
    tn = np.zeros(trace_n, dtype=np.int64)

    for p in range(n_part):
        x = np.random.random() * W
        z = 0.0
        a = _launch_dir(np.random.random())
        dx, dz = np.sin(a), np.cos(a)
        rec = p < trace_n
        k = 0
        if rec:
            tx[p, 0], tz[p, 0] = x, z
            k = 1

        for _bounce in range(2000):
            t_best = 1e18
            wall = -1                       # 0:x=0  1:x=W  2:bottom  3:escape
            if dx < -1e-12:
                t = -x / dx
                if t < t_best:
                    t_best, wall = t, 0
            if dx > 1e-12:
                t = (W - x) / dx
                if t < t_best:
                    t_best, wall = t, 1
            if dz > 1e-12:
                t = (H - z) / dz
                if t < t_best:
                    t_best, wall = t, 2
            if dz < -1e-12:
                t = -z / dz
                if t < t_best:
                    t_best, wall = t, 3

            x += dx * t_best
            z += dz * t_best
            if rec and k < 200:
                tx[p, k], tz[p, k] = x, z
                k += 1

            if wall == 3:                   # left through the opening
                break

            ib = int(z / H * NBIN)
            if ib >= NBIN:
                ib = NBIN - 1
            if ib < 0:
                ib = 0

            s = s0 * (1.0 - theta[ib]) ** n_steric      # self-limiting
            if np.random.random() < s:
                theta[ib] += weight / sites_per_bin
                if theta[ib] > 1.0:
                    theta[ib] = 1.0
                break

            if np.random.random() > reemit:             # non-ideal loss
                break

            a = _launch_dir(np.random.random())
            if wall == 0:
                dx, dz = np.cos(a), np.sin(a)
            elif wall == 1:
                dx, dz = -np.cos(a), np.sin(a)
            else:
                dz, dx = -np.cos(a), np.sin(a)

        if rec:
            tn[p] = k
    return theta, tx, tz, tn


def profile(AR, dose, s0, n_steric=1.0, reemit=1.0, sites_per_bin=400.0,
            weight=1.0, trace_n=0, seed=1):
    """`dose` = molecules entering per unit opening area (exposure)."""
    n_part = int(dose / weight)
    return run_mc(AR, n_part, s0, n_steric, reemit, sites_per_bin, weight,
                  max(trace_n, 1), seed)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    AR = 20.0
    SPB = 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 11.2))
    fig.subplots_adjust(hspace=0.34, wspace=0.30)

    # (a) geometry and discretisation ---------------------------------
    ax = axes[0, 0]
    ax.add_patch(plt.Rectangle((-1.1, 0), 1.1, AR, color="#c9cdd4"))
    ax.add_patch(plt.Rectangle((1.0, 0), 1.1, AR, color="#c9cdd4"))
    ax.add_patch(plt.Rectangle((-1.1, AR), 3.2, 2.2, color="#c9cdd4"))
    for i in range(NBIN + 1):
        zz = i / NBIN * AR
        ax.plot([-0.17, 0.0], [zz, zz], lw=0.5, color="#d33")
        ax.plot([1.0, 1.17], [zz, zz], lw=0.5, color="#d33")
    ax.annotate("", xy=(0, -1.5), xytext=(1, -1.5),
                arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(1.35, -1.5, "W", ha="left", va="center", fontsize=12)
    ax.annotate("", xy=(2.35, 0), xytext=(2.35, AR),
                arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(2.55, AR / 2, "H", va="center", fontsize=12)
    for xx in (0.15, 0.5, 0.85):
        ax.annotate("", xy=(xx, -0.2), xytext=(xx, -4.0),
                    arrowprops=dict(arrowstyle="->", color="#178", lw=1.8))
    ax.text(0.5, -5.0, "Precursor influx", ha="center", fontsize=11.5,
            color="#178")
    ax.text(-0.32, AR * 0.55,
            "Sidewall split into\n64 depth bins\none coverage $\\theta_i$ each",
            ha="right", va="center", fontsize=11, color="#d33")
    ax.set_xlim(-4.6, 3.1)
    ax.set_ylim(-6.2, AR * 1.14)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("(a) Geometry and discretisation\n"
                 "AR = H/W = 20, width is not binned", fontsize=13)

    # (b) trajectories ------------------------------------------------
    ax = axes[0, 1]
    _, tx, tz, tn = profile(AR, 4000, s0=0.006, trace_n=800, seed=11)
    ax.add_patch(plt.Rectangle((-0.6, 0), 0.6, AR, color="#e6e8ec"))
    ax.add_patch(plt.Rectangle((1.0, 0), 0.6, AR, color="#e6e8ec"))
    ax.add_patch(plt.Rectangle((-0.6, AR), 2.2, 1.4, color="#e6e8ec"))
    picks, cols = [], ["#c0392b", "#2874a6", "#1e8449"]
    depths = [(p, tz[p, tn[p] - 1]) for p in range(800) if tn[p] > 3]
    for lo, hi in [(0.0, 0.25), (0.3, 0.6), (0.65, 1.0)]:
        for p, d in depths:
            if lo * AR <= d <= hi * AR and 5 <= tn[p] <= 22:
                picks.append(p)
                break
    for j, p in enumerate(picks[:3]):
        k = tn[p]
        ax.plot(tx[p, :k], tz[p, :k], lw=1.3, alpha=0.9, color=cols[j])
        ax.plot(tx[p, k - 1], tz[p, k - 1], "o", ms=8, color=cols[j], zorder=5)
    ax.set_xlim(-0.9, 2.0)
    ax.set_ylim(-2.0, AR * 1.10)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("(b) Ballistic MC — life of one particle\n"
                 "fly straight, hit wall, react with $s(\\theta)$ or re-emit",
                 fontsize=13)
    ax.text(0.5, -1.3, "$\\bullet$  where it reacted and stopped",
            ha="center", fontsize=11)

    zc = (np.arange(NBIN) + 0.5) / NBIN

    # (c) hidden parameters -------------------------------------------
    ax = axes[1, 0]
    for v, c in [(0.002, "#2874a6"), (0.010, "#1e8449"), (0.050, "#c0392b")]:
        th, _, _, _ = profile(AR, 100_000, s0=v, sites_per_bin=SPB, seed=3)
        ax.plot(th[:-1], zc[:-1], lw=2.3, color=c, label=f"$s_0$ = {v:.3f}")
    ax.set_xlabel(r"Coverage  $\theta$   (= thickness / saturation thickness)",
                  fontsize=12)
    ax.set_ylabel("Normalized depth  z / H   (sidewall)", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_title("(c) Effect of the hidden parameter — to be inferred\n"
                 "same dose and AR, only $s_0$ differs", fontsize=13)

    # (d) process conditions ------------------------------------------
    ax = axes[1, 1]
    for dn, lab, c in [(40_000, r"dose $\times$0.6", "#8e44ad"),
                       (100_000, r"dose $\times$1.6", "#b9770e"),
                       (300_000, r"dose $\times$4.7", "#117a65")]:
        th, _, _, _ = profile(AR, dn, s0=0.010, sites_per_bin=SPB, seed=3)
        ax.plot(th[:-1], zc[:-1], lw=2.3, color=c, label=lab)
    ax.set_xlabel(r"Coverage  $\theta$", fontsize=12)
    ax.set_ylabel("Normalized depth  z / H   (sidewall)", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_title("(d) Effect of the process condition — known input\n"
                 "same $s_0$ and AR, only dose differs", fontsize=13)

    fig.suptitle("What the feature-scale MC simulator does — AR 20 trench",
                 fontsize=15.5, y=0.965)
    fig.savefig("trench_mc_explained.png", dpi=155, bbox_inches="tight",
                facecolor="white")
    print("saved")
