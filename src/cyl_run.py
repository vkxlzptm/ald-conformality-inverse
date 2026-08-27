"""Dose-checkpointed cylinder MC.

One MC run is inherently sequential in dose, so a single run can be snapshotted at
several dose levels and every snapshot is a valid training example.  The run also
stops adaptively once the deepest bin saturates, so no dose is wasted, and a hard
particle budget bounds the worst-case cost.
"""
import numpy as np
from numba import njit
from cyl_mc import NBIN, R, _cos_dir_local, _wall_t


@njit(cache=True)
def run_ckpt(AR, s0, n_steric, reemit, sites_per_bin, weight,
             ckpt, max_part, stop_theta, seed):
    """
    ckpt : (K,) int64, ascending pseudo-particle counts at which to snapshot.

    Returns
    -------
    snaps  : (K, NBIN) coverage at each checkpoint
    valid  : (K,) 1 if that checkpoint was reached
    n_done : pseudo-particles actually launched
    """
    np.random.seed(seed)
    H = AR * 2.0 * R
    K = ckpt.shape[0]
    theta = np.zeros(NBIN)
    snaps = np.zeros((K, NBIN))
    valid = np.zeros(K, dtype=np.int64)

    kc = 0
    p = 0
    while p < max_part:
        rr = R * np.sqrt(np.random.random())
        ph = 2.0 * np.pi * np.random.random()
        x, y, z = rr * np.cos(ph), rr * np.sin(ph), 0.0
        dz, dx, dy = _cos_dir_local(np.random.random(), np.random.random())

        for _b in range(200000):
            tw = _wall_t(x, y, dx, dy, R)
            tz = 1e18
            hit_end = 0
            if dz > 1e-12:
                tz = (H - z) / dz
                hit_end = 2
            elif dz < -1e-12:
                tz = -z / dz
                hit_end = 3
            bottom = False
            if tz < tw:
                z += dz * tz
                x += dx * tz
                y += dy * tz
                if hit_end == 3:
                    break
                ib = NBIN - 1
                bottom = True
            else:
                x += dx * tw
                y += dy * tw
                z += dz * tw
                ib = int(z / H * NBIN)
                if ib >= NBIN:
                    ib = NBIN - 1
                if ib < 0:
                    ib = 0

            s = s0 * (1.0 - theta[ib]) ** n_steric
            if s > 0.0 and np.random.random() < s:
                theta[ib] += weight / sites_per_bin
                if theta[ib] > 1.0:
                    theta[ib] = 1.0
                break
            if reemit < 1.0 and np.random.random() > reemit:
                break

            cn, t1, t2 = _cos_dir_local(np.random.random(), np.random.random())
            if bottom:
                dz = -cn
                dx, dy = t1, t2
            else:
                nx, ny = -x / R, -y / R
                dx = cn * nx - t1 * ny
                dy = cn * ny + t1 * nx
                dz = t2

        p += 1
        while kc < K and p >= ckpt[kc]:
            snaps[kc] = theta
            valid[kc] = 1
            kc += 1
        if theta[NBIN - 1] >= stop_theta:
            break

    # fill any remaining checkpoints with the saturated end state
    while kc < K:
        snaps[kc] = theta
        valid[kc] = 2                 # 2 = reached only because the run saturated
        kc += 1
    return snaps, valid, p
