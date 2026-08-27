"""Least-squares baseline: the conventional way this inverse problem is solved today.

For each measured profile set, run a derivative-free optimiser that calls the MC
simulator over and over until the simulated profiles match the measurement.  This
is what the amortized network has to beat, so it is given a fair fight: a
transformed (unbounded) search space, multi-start, a stated evaluation budget, and
common random numbers to tame the stochastic objective.

What it reports, per case: recovered parameters, error against ground truth,
simulator calls, and wall-clock time.

    python src/baseline_ls.py --cases 4 --budget 400 --starts 2
"""
import argparse
import time

import numpy as np
from scipy.optimize import minimize

from cyl_mc import NBIN
from cyl_run import run_ckpt
import generate_dataset as G

BENCH_SHARD = 900          # reserved: never used for training data (0..599)
OBS_CKPT = 3               # which dose checkpoint is "the measurement"
MEAS_NOISE = 0.03          # relative measurement noise, 1 sigma
N_MASKED = 10              # depth bins unavailable in the measurement
CRN_SEED = 4242            # common random numbers: fixed across all evaluations

# search box = the generating prior, widened slightly so the truth is interior
BOX = dict(s0_ref=(0.002, 0.8), Ea=(0.0, 80.0), n_steric=(0.0, 5.0),
           reemit=(0.85, 1.0), n_sites=(1.5, 8.0))
LOGPAR = {"s0_ref", "n_sites"}
NAMES = ["s0_ref", "Ea", "n_steric", "reemit", "n_sites"]


# ------------------------------------------------------- parameter transform
def _to_unit(name, v):
    lo, hi = BOX[name]
    if name in LOGPAR:
        lo, hi, v = np.log(lo), np.log(hi), np.log(v)
    return (v - lo) / (hi - lo)


def _from_unit(name, u):
    lo, hi = BOX[name]
    if name in LOGPAR:
        lo, hi = np.log(lo), np.log(hi)
    v = lo + u * (hi - lo)
    return np.exp(v) if name in LOGPAR else v


def pack(p):
    """physical dict -> unbounded vector (logit of the unit box)."""
    u = np.clip([_to_unit(n, p[n]) for n in NAMES], 1e-4, 1 - 1e-4)
    return np.log(u / (1 - u))


def unpack(x):
    u = 1.0 / (1.0 + np.exp(-np.asarray(x, float)))
    return {n: _from_unit(n, ui) for n, ui in zip(NAMES, u)}


# ------------------------------------------------------------ forward model
def forward(p, AR, pi2, seed):
    """Three temperature profiles at one dose level. One 'simulator call'."""
    spb = p["n_sites"] * np.pi * AR / NBIN * G.SPB_SCALE
    weight = max(1.0, spb / G.NOISE_TARGET)
    ckpt = np.array([max(1, int(pi2 * spb * NBIN / weight))], dtype=np.int64)
    out = np.zeros((3, NBIN))
    for j, T in enumerate(G.TEMPS_K):
        s0 = float(G.s0_of_T(p["s0_ref"], p["Ea"], T))
        snaps, _, _ = run_ckpt(AR, s0, p["n_steric"], p["reemit"], spb, weight,
                               ckpt, int(ckpt[-1]), G.STOP_THETA, seed + j)
        out[j] = snaps[0]
    return out


# ------------------------------------------------------------------- a case
def make_case(idx):
    """Reproducible benchmark case: truth, noisy masked observation, conditions."""
    rng = np.random.default_rng(1_000_003 + BENCH_SHARD + idx)
    truth = G.draw_params(rng)
    pi2 = float(G.PI2_CKPT[OBS_CKPT])
    clean = forward(truth, truth["AR"], pi2, seed=BENCH_SHARD * 100_003 + idx * 3)

    orng = np.random.default_rng(77_000 + idx)
    obs = clean * (1.0 + MEAS_NOISE * orng.standard_normal(clean.shape))
    obs = np.clip(obs, 0.0, 1.0)
    mask = np.ones(NBIN, bool)
    mask[orng.choice(NBIN, N_MASKED, replace=False)] = False
    return truth, obs, mask, truth["AR"], pi2


def sse(p, obs, mask, AR, pi2):
    sim = forward(p, AR, pi2, CRN_SEED)          # common random numbers
    return float(((sim[:, mask] - obs[:, mask]) ** 2).sum())


# ------------------------------------------------------------------- the fit
def fit(obs, mask, AR, pi2, budget, starts, method, verbose=False):
    calls = {"n": 0}

    def obj(x):
        calls["n"] += 1
        return sse(unpack(x), obs, mask, AR, pi2)

    rng = np.random.default_rng(31337)
    best, best_f = None, np.inf
    t0 = time.time()
    for k in range(starts):
        x0 = rng.normal(0.0, 1.0, len(NAMES)) if k else np.zeros(len(NAMES))
        r = minimize(obj, x0, method=method,
                     options=dict(maxfev=budget // starts, xatol=1e-3,
                                  fatol=1e-6, adaptive=True))
        if verbose:
            print(f"      start {k}: sse {r.fun:.4f} after {calls['n']} calls",
                  flush=True)
        if r.fun < best_f:
            best, best_f = r.x, r.fun
    return unpack(best), best_f, calls["n"], time.time() - t0


def rel_err(est, truth):
    out = {}
    for n in NAMES:
        if n == "Ea":                              # additive: absolute, kJ/mol
            out[n] = abs(est[n] - truth[n])
        else:
            out[n] = abs(est[n] - truth[n]) / max(abs(truth[n]), 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--budget", type=int, default=400,
                    help="simulator calls per case, across all starts")
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--method", default="Nelder-Mead")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    # warm the JIT so it is not charged to the first case
    run_ckpt(10.0, 0.05, 1.0, 1.0, 400.0, 1.0,
             np.array([50], dtype=np.int64), 50, 0.999, 1)

    print(f"method {a.method}   budget {a.budget} calls/case   "
          f"{a.starts} starts   benchmark shard {BENCH_SHARD}\n")
    rows = []
    for i in range(a.cases):
        truth, obs, mask, AR, pi2 = make_case(i)
        pi1 = AR * np.sqrt(truth["s0_ref"])
        print(f"case {i}:  AR {AR:5.1f}  Pi1 {pi1:5.2f}  "
              f"s0_ref {truth['s0_ref']:.4f}  Ea {truth['Ea']:5.1f}", flush=True)
        est, f, n, dt = fit(obs, mask, AR, pi2, a.budget, a.starts, a.method,
                            a.verbose)
        e = rel_err(est, truth)
        rows.append((n, dt, e))
        print(f"          -> {n} calls, {dt/60:.1f} min, sse {f:.4f}")
        print("          " + "  ".join(
            f"{k} {est[k]:.3f}/{truth[k]:.3f}" for k in NAMES))
        print("          err: " + "  ".join(
            f"{k} {e[k]*100:.0f}%" if k != "Ea" else f"Ea {e[k]:.1f}kJ"
            for k in NAMES), flush=True)

    print("\n=== baseline summary ===")
    print(f"  simulator calls per case : {np.mean([r[0] for r in rows]):.0f}")
    print(f"  wall clock per case      : {np.mean([r[1] for r in rows])/60:.1f} min")
    for k in NAMES:
        v = [r[2][k] for r in rows]
        unit = "kJ/mol" if k == "Ea" else "%"
        scale = 1.0 if k == "Ea" else 100.0
        print(f"  median error {k:9s}    : {np.median(v)*scale:.1f} {unit}")
    print("  uncertainty              : none (point estimate only)")


if __name__ == "__main__":
    main()
