"""Least-squares baseline: how this inverse problem is solved today.

For one measured profile, run a derivative-free optimiser that calls the MC
simulator until the simulated profile matches.  This is what the amortized
network has to beat, so it gets a fair fight: an unbounded search space, multi
start, an explicit evaluation budget, and common random numbers to tame the
stochastic objective.

Unknowns are the same four the network predicts, all dimensionless:
    s0, steric exponent n, re-emission survival, Pi2 = dose per surface site.

`sites_per_bin` is NOT one of them.  The profile depends on dose and site count
only through Pi2, so the absolute site count is a pure numerical resolution
knob here, fixed to SPB_REF, and it sets the Monte Carlo noise and nothing else.

    python src/baseline_ls.py --cases 8 --budget 1500 --starts 2 --workers 8
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

from cyl_mc import NBIN
from cyl_run import run_ckpt
import data as D
import generate_dataset as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_SHARD = 900          # reserved: never used for training data
MEAS_NOISE = D.MEAS_NOISE
N_MASKED = D.N_MASKED
CRN_SEED = 4242            # common random numbers: fixed across all evaluations
SIMPLEX_STEP = 2.0         # initial simplex edge, in logit units
SPB_REF = 2000.0           # numerical resolution only; cancels from the physics

NAMES = D.TARGETS          # s0, n_steric, reemit, pi2
BOX = dict(s0=(0.002, 0.9), n_steric=(0.0, 5.0), reemit=(0.85, 1.0),
           pi2=(0.4, 60.0))
LOGPAR = {"s0", "pi2"}


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
    u = np.clip([_to_unit(n, p[n]) for n in NAMES], 1e-4, 1 - 1e-4)
    return np.log(u / (1 - u))


def unpack(x):
    u = 1.0 / (1.0 + np.exp(-np.asarray(x, float)))
    return {n: _from_unit(n, ui) for n, ui in zip(NAMES, u)}


# ------------------------------------------------------------ forward model
def forward(p, AR, seed):
    """One profile at one temperature. Counts as one simulator call."""
    weight = max(1.0, SPB_REF / G.NOISE_TARGET)
    ck = np.array([max(1, int(p["pi2"] * SPB_REF * NBIN / weight))], np.int64)
    snaps, _, _ = run_ckpt(AR, p["s0"], p["n_steric"], p["reemit"], SPB_REF,
                           weight, ck, int(ck[-1]), G.STOP_THETA, seed)
    return snaps[0]


def make_case(idx, AR=None):
    """Reproducible benchmark case, drawn from the same prior as the training set.

    `AR` overrides the drawn aspect ratio without touching the random stream, so
    the transfer study (src/transfer_ar.py) can put every case at one AR and still
    reproduce the cases used here.
    """
    rng = np.random.default_rng(1_000_003 + BENCH_SHARD + idx)
    q = G.draw_params(rng)
    if AR is not None:
        q["AR"] = float(AR)
    pi2 = float(G.PI2_CKPT[3] * np.exp(rng.uniform(-G.PI2_JITTER, G.PI2_JITTER)))
    T = float(G.TEMPS_K[rng.integers(len(G.TEMPS_K))])
    truth = dict(s0=float(G.s0_of_T(q["s0_ref"], q["Ea"], T)),
                 n_steric=q["n_steric"], reemit=q["reemit"], pi2=pi2)
    clean = forward(truth, q["AR"], seed=BENCH_SHARD * 100_003 + idx)

    orng = np.random.default_rng(77_000 + idx)
    obs = np.clip(clean * (1 + MEAS_NOISE * orng.standard_normal(clean.shape)),
                  0.0, 1.0)
    mask = np.ones(NBIN, bool)
    mask[orng.choice(NBIN, N_MASKED, replace=False)] = False
    return truth, obs, mask, q["AR"], T


def sse(p, obs, mask, AR):
    sim = forward(p, AR, CRN_SEED)              # common random numbers
    return float(((sim[mask] - obs[mask]) ** 2).sum())


# ------------------------------------------------------------------- the fit
def fit(obs, mask, AR, budget, starts, method, verbose=False):
    calls = {"n": 0}

    def obj(x):
        calls["n"] += 1
        return sse(unpack(x), obs, mask, AR)

    rng = np.random.default_rng(31337)
    best, best_f = None, np.inf
    n = len(NAMES)
    t0 = time.time()
    for k in range(starts):
        x0 = rng.normal(0.0, 1.2, n) if k else np.zeros(n)
        # Explicit initial simplex.  scipy perturbs each coordinate by 5 % and
        # falls back to 0.00025 where x0 is exactly zero, which makes the start
        # degenerate and stalls the search after ~150 calls.  Measured.
        sim0 = np.vstack([x0] + [x0 + SIMPLEX_STEP * np.eye(n)[i]
                                 for i in range(n)])
        r = minimize(obj, x0, method=method,
                     options=dict(maxfev=budget // starts, xatol=1e-3,
                                  fatol=1e-6, adaptive=True,
                                  initial_simplex=sim0))
        if verbose:
            print(f"      start {k}: sse {r.fun:.4f} after {calls['n']} calls",
                  flush=True)
        if r.fun < best_f:
            best, best_f = r.x, r.fun
    return unpack(best), best_f, calls["n"], time.time() - t0


def rel_err(est, truth):
    out = {}
    for n in NAMES:
        if n == "reemit":                       # value sits near 1; use absolute
            out[n] = abs(est[n] - truth[n])
        else:
            out[n] = abs(est[n] - truth[n]) / max(abs(truth[n]), 1e-9)
    return out


_JOB = {}


def _worker_init(budget, starts, method):
    """macOS and Windows spawn a fresh interpreter per worker: globals set in the
    parent and a JIT warm-up done there do not carry over, so both happen here."""
    _JOB.update(budget=budget, starts=starts, method=method)
    run_ckpt(10.0, 0.05, 1.0, 1.0, 400.0, 1.0,
             np.array([50], dtype=np.int64), 50, 0.999, 1)


def solve_case(i):
    truth, obs, mask, AR, T = make_case(i)
    est, f, n, dt = fit(obs, mask, AR, _JOB["budget"], _JOB["starts"],
                        _JOB["method"], False)
    return i, truth, est, f, n, dt, AR, T


def _report(i, truth, est, f, n, dt, AR, T):
    e = rel_err(est, truth)
    pi1 = AR * np.sqrt(truth["s0"])
    print(f"case {i}:  AR {AR:5.1f}  Pi1 {pi1:5.2f}  T {T-273.15:5.0f} C   "
          f"-> {n} calls, {dt/60:.1f} min, sse {f:.4f}")
    print("          " + "  ".join(
        f"{k} {est[k]:.3f}/{truth[k]:.3f}" for k in NAMES))
    print("          err: " + "  ".join(
        f"{k} {e[k]*100:.0f}%" if k != "reemit" else f"reemit {e[k]:.3f}"
        for k in NAMES), flush=True)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--method", default="Nelder-Mead")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=os.path.join(ROOT, "results",
                                                   "baseline.json"))
    a = ap.parse_args()

    _JOB.update(budget=a.budget, starts=a.starts, method=a.method)
    run_ckpt(10.0, 0.05, 1.0, 1.0, 400.0, 1.0,
             np.array([50], dtype=np.int64), 50, 0.999, 1)

    workers = min(a.workers or max(1, (os.cpu_count() or 2) - 2), a.cases)
    print(f"method {a.method}   budget {a.budget} calls/case   {a.starts} starts"
          f"   benchmark shard {BENCH_SHARD}   {workers} workers\n", flush=True)

    rows, recs, t_wall = [], [], time.time()

    def keep(r):
        rows.append((r[4], r[5], _report(*r)))
        recs.append(dict(case=r[0], truth={k: float(r[1][k]) for k in NAMES},
                         est={k: float(r[2][k]) for k in NAMES},
                         AR=float(r[6]), T_K=float(r[7]), sse=float(r[3]),
                         calls=int(r[4]), seconds=float(r[5])))

    if workers == 1:
        for i in range(a.cases):
            keep(solve_case(i))
    else:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_worker_init,
                     initargs=(a.budget, a.starts, a.method)) as pool:
            for r in pool.imap_unordered(solve_case, range(a.cases)):
                keep(r)

    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as fh:
        json.dump(dict(method=a.method, budget=a.budget, starts=a.starts,
                       bench_shard=BENCH_SHARD,
                       cases=sorted(recs, key=lambda d: d["case"])), fh, indent=1)

    print(f"\n  total wall clock: {(time.time()-t_wall)/60:.1f} min "
          f"({workers} cases at a time)")
    print("\n=== baseline summary ===")
    print(f"  simulator calls per case : {np.mean([r[0] for r in rows]):.0f}")
    print(f"  wall clock per case      : {np.mean([r[1] for r in rows])/60:.1f} min")
    for k in NAMES:
        v = [r[2][k] for r in rows]
        if k == "reemit":
            print(f"  median error {k:9s}    : {np.median(v):.3f} (absolute)")
        else:
            print(f"  median error {k:9s}    : {np.median(v)*100:.1f} %")
    print("  uncertainty              : none (point estimate only)")
    print(f"  wrote {a.json}")


if __name__ == "__main__":
    main()
