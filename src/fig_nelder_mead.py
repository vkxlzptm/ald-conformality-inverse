"""Explanatory figure: what the least-squares baseline actually does.

(a) the simplex moves, schematically
(b) the REAL sum-of-squares landscape of this problem, on a 2D slice, with the
    actual Nelder-Mead path drawn on it
(c) how slowly the error falls per simulator call
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import baseline_ls as B
import generate_dataset as G
from cyl_run import run_ckpt

# ----------------------------------------------------------------- the case
TRUTH = dict(AR=15.0, s0_ref=0.05, Ea=30.0, n_steric=2.0, reemit=0.97,
             n_sites=4.0)
PI2 = float(G.PI2_CKPT[B.OBS_CKPT])

run_ckpt(10., .05, 1., 1., 400., 1., np.array([50], np.int64), 50, .999, 1)

clean = B.forward(TRUTH, TRUTH["AR"], PI2, seed=555)
orng = np.random.default_rng(9)
OBS = np.clip(clean * (1 + B.MEAS_NOISE * orng.standard_normal(clean.shape)), 0, 1)
MASK = np.ones(B.NBIN, bool)
MASK[orng.choice(B.NBIN, B.N_MASKED, replace=False)] = False

S0_RANGE = np.geomspace(0.012, 0.20, 22)
NS_RANGE = np.linspace(0.2, 4.2, 22)


def sse2(ls0, nst):
    p = dict(TRUTH); p["s0_ref"] = float(np.exp(ls0)); p["n_steric"] = float(nst)
    return B.sse(p, OBS, MASK, TRUTH["AR"], PI2)


t0 = time.time()
Z = np.empty((len(NS_RANGE), len(S0_RANGE)))
for i, nst in enumerate(NS_RANGE):
    for j, s0 in enumerate(S0_RANGE):
        Z[i, j] = sse2(np.log(s0), nst)
    print(f"  landscape row {i+1}/{len(NS_RANGE)}  ({time.time()-t0:.0f}s)",
          flush=True)

path, hist = [], []
def obj2(v):
    f = sse2(v[0], v[1])
    path.append(v.copy()); hist.append(f)
    return f

x0 = np.array([np.log(0.16), 0.6])
minimize(obj2, x0, method="Nelder-Mead",
         options=dict(maxfev=140, xatol=1e-4, fatol=1e-8, adaptive=True))
path = np.array(path)
best = np.minimum.accumulate(hist)
np.savez("/tmp/w/nm_fig_data.npz", Z=Z, S0=S0_RANGE, NS=NS_RANGE,
         path=path, hist=hist, best=best)
print("landscape + path done", time.time()-t0, flush=True)
