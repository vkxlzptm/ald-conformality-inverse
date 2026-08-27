"""3D axisymmetric cylindrical via: ballistic (free-molecular) Monte Carlo.

Knudsen regime: no gas-gas collisions, so absolute size is irrelevant and only
AR = H/D matters.  D = 2R is fixed to 1, H = AR.

Notation follows Cremers, Puurunen & Dendooven, Appl. Phys. Rev. 6, 021302 (2019):
    s  = sticking probability per wall collision,  s0 = value on the bare surface,
    s(theta) = s0 (1 - theta)^n_steric.
"""
import numpy as np
from numba import njit

NBIN = 64
R = 0.5


@njit(cache=True)
def _cos_dir_local(u1, u2):
    """Cosine (Knudsen) law about +n: returns (c_n, c_t1, c_t2)."""
    c = np.sqrt(u1)                     # cos(polar) ~ sqrt(U) for p ~ cos*sin
    st = np.sqrt(1.0 - u1)
    ph = 2.0 * np.pi * u2
    return c, st * np.cos(ph), st * np.sin(ph)


@njit(cache=True)
def _wall_t(x, y, dx, dy, R):
    """Positive root of |P + t d|_xy^2 = R^2 for P already on the wall (or inside)."""
    a = dx * dx + dy * dy
    if a < 1e-18:
        return 1e18
    b = x * dx + y * dy
    c = x * x + y * y - R * R
    disc = b * b - a * c
    if disc < 0.0:
        return 1e18
    sq = np.sqrt(disc)
    t = (-b + sq) / a
    if t <= 1e-12:
        return 1e18
    return t


@njit(cache=True)
def run_cyl(AR, n_part, s0, n_steric, reemit, sites_per_bin, weight,
            open_bottom, seed):
    """
    open_bottom = 1 : tube open at z = H, count crossings  -> transmission test
    open_bottom = 0 : closed via, bottom is a reacting surface

    Returns (theta[NBIN], transmitted, n_collisions)
    """
    np.random.seed(seed)
    H = AR * 2.0 * R
    theta = np.zeros(NBIN)
    trans = 0
    ncoll = 0

    for _p in range(n_part):
        # enter through the opening disk at z = 0, cosine about +z
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
                hit_end = 2                      # bottom / exit plane
            elif dz < -1e-12:
                tz = -z / dz
                hit_end = 3                      # back out of the opening
            if tz < tw:
                z += dz * tz
                x += dx * tz
                y += dy * tz
                if hit_end == 3:
                    break
                if open_bottom == 1:
                    trans += 1
                    break
                ib = NBIN - 1                    # closed bottom -> deepest bin
            else:
                x += dx * tw
                y += dy * tw
                z += dz * tw
                ib = int(z / H * NBIN)
                if ib >= NBIN:
                    ib = NBIN - 1
                if ib < 0:
                    ib = 0
            ncoll += 1

            s = s0 * (1.0 - theta[ib]) ** n_steric
            if s > 0.0 and np.random.random() < s:
                theta[ib] += weight / sites_per_bin
                if theta[ib] > 1.0:
                    theta[ib] = 1.0
                break
            if reemit < 1.0 and np.random.random() > reemit:
                break

            # diffuse re-emission about the inward normal
            cn, t1, t2 = _cos_dir_local(np.random.random(), np.random.random())
            if tz < tw:                          # bottom: inward normal is -z
                dz = -cn
                dx, dy = t1, t2
            else:                                # side wall: inward normal -r_hat
                nx, ny = -x / R, -y / R
                dx = cn * nx - t1 * ny
                dy = cn * ny + t1 * nx
                dz = t2
    return theta, trans, ncoll


def transmission(AR, n_part=200_000, seed=5):
    _, tr, nc = run_cyl(AR, n_part, 0.0, 1.0, 1.0, 1.0, 1.0, 1, seed)
    return tr / n_part, nc / n_part


def profile(AR, dose, s0, n_steric=1.0, reemit=1.0, sites_per_bin=400.0,
            weight=1.0, seed=1):
    """`dose` = molecules entering per unit opening area (exposure)."""
    n_part = int(dose / weight)
    th, _, nc = run_cyl(AR, n_part, s0, n_steric, reemit, sites_per_bin,
                        weight, 0, seed)
    return th, nc / max(n_part, 1)
