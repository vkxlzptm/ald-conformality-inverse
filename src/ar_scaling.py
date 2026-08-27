"""AR scaling measurements: (i) zero-reaction transmission vs AR, (ii) saturation dose vs AR.

Notation: s0 = initial sticking probability (Cremers et al., Appl. Phys. Rev. 6, 021302).
"""
import numpy as np
from numba import njit

NBIN = 64


@njit(cache=True)
def _dir(u):
    return np.arcsin(2.0 * u - 1.0)


@njit(cache=True)
def transmission(AR, n_part, seed):
    """s0 = 0, purely diffuse re-emission. Fraction of particles reaching the bottom."""
    np.random.seed(seed)
    H, W = AR, 1.0
    hit = 0
    for _ in range(n_part):
        x = np.random.random() * W
        z = 0.0
        a = _dir(np.random.random())
        dx, dz = np.sin(a), np.cos(a)
        for _b in range(200000):
            t_best = 1e18
            wall = -1
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
            if wall == 3:
                break
            if wall == 2:
                hit += 1
                break
            a = _dir(np.random.random())
            if wall == 0:
                dx, dz = np.cos(a), np.sin(a)
            else:
                dx, dz = -np.cos(a), np.sin(a)
    return hit / n_part


@njit(cache=True)
def dose_to_saturate(AR, s0, sites_per_bin, target_bin, thresh, n_max, seed):
    """Particles needed until bin `target_bin` reaches coverage `thresh`."""
    np.random.seed(seed)
    H, W = AR, 1.0
    theta = np.zeros(NBIN)
    for p in range(n_max):
        x = np.random.random() * W
        z = 0.0
        a = _dir(np.random.random())
        dx, dz = np.sin(a), np.cos(a)
        for _b in range(200000):
            t_best = 1e18
            wall = -1
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
            if wall == 3:
                break
            ib = int(z / H * NBIN)
            if ib >= NBIN:
                ib = NBIN - 1
            s = s0 * (1.0 - theta[ib])
            if np.random.random() < s:
                theta[ib] += 1.0 / sites_per_bin
                if theta[ib] > 1.0:
                    theta[ib] = 1.0
                break
            a = _dir(np.random.random())
            if wall == 0:
                dx, dz = np.cos(a), np.sin(a)
            elif wall == 1:
                dx, dz = -np.cos(a), np.sin(a)
            else:
                dz, dx = -np.cos(a), np.sin(a)
        if theta[target_bin] >= thresh:
            return p + 1
    return -1


if __name__ == "__main__":
    print("=== (i) transmission at s0 = 0 ===")
    ARs = np.array([2.0, 5.0, 10.0, 20.0, 40.0])
    tr = np.array([transmission(a, 200_000, 5) for a in ARs])
    for a, t in zip(ARs, tr):
        print(f"  AR={a:5.0f}   transmission={t:.4f}   1/AR={1/a:.4f}")
    sl = np.polyfit(np.log(ARs[1:]), np.log(tr[1:]), 1)[0]
    print(f"  -> log-log slope (AR>=5): {sl:.3f}")

    print("\n=== (ii) saturation dose vs AR (s0=0.02, constant site density) ===")
    print("    target: bin at z/H = 0.94 reaching theta = 0.90")
    res = []
    for a in [5.0, 10.0, 20.0, 40.0]:
        spb = 40.0 * a                  # bin height ~ AR, so site density is constant
        n = dose_to_saturate(a, 0.02, spb, 60, 0.90, 200_000_000, 11)
        res.append((a, n, n / spb))
        print(f"  AR={a:5.0f}   particles needed={n:>12,}   (per site {n/spb/64:8.1f})")
    A = np.array([r[0] for r in res], dtype=float)
    N = np.array([r[1] for r in res], dtype=float)
    sl2 = np.polyfit(np.log(A), np.log(N), 1)[0]
    print(f"  -> log-log slope: {sl2:.3f}   (ideal asymptote 2.0)")
