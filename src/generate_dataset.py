"""Training-set generation for the ALD conformality inverse problem.

Geometry   : 3D axisymmetric cylindrical via (see src/cyl_mc.py), AR = H/D.
Observation: one training example is a *temperature series* -- the same recipe run
             at three temperatures, giving three depth profiles.  A single profile
             cannot identify an activation energy, because T then enters only
             through s0(T) and is a pure reparametrisation; a series can.
Targets    : s0_ref, Ea, n_steric, reemit, n_sites.

Cost control
------------
An MC run is sequential in dose, so one run is snapshotted at K dose levels and
every snapshot is a valid example.  One parameter draw = 3 runs (3 temperatures)
= K examples.  Measured: about 7.6 s per example on one core.

Checkpointing
-------------
Shard files are written once and skipped if present, so re-running the script is
the same as resuming.  Seeds are derived from (shard, index) so a shard is
reproducible on its own.

Usage
-----
    python src/generate_dataset.py                 # all shards, ncpu workers
    python src/generate_dataset.py --shards 0-7    # a subset
    python src/generate_dataset.py --dry-run       # cost estimate only
"""
import argparse
import os
import time

import numpy as np

from cyl_mc import NBIN
from cyl_run import run_ckpt

# ---------------------------------------------------------------- configuration
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "dataset")

N_SHARD = 60                 # 10 per core on a 6-core machine
DRAWS_PER_SHARD = 30         # 60 * 30 * 6 = 10,800 examples

TEMPS_K = np.array([473.15, 523.15, 573.15])     # 200, 250, 300 C
T_REF = 523.15
R_GAS = 8.314462618e-3                            # kJ/(mol K)

K_CKPT = 6
PI2_CKPT = np.geomspace(0.8, 30.0, K_CKPT)        # dose per surface site

PRIOR = dict(
    AR=(10.0, 50.0),          # log-uniform,  H / D
    s0_ref=(0.005, 0.5),      # log-uniform,  sticking probability at T_REF
    Ea=(0.0, 60.0),           # uniform,      kJ/mol
    n_steric=(0.0, 4.0),      # uniform,      s = s0 (1 - theta)^n
    reemit=(0.90, 1.0),       # uniform,      re-emission survival probability
    n_sites=(2.0, 6.0),       # log-uniform,  arbitrary areal site density units
)

SPB_SCALE = 700.0            # sites_per_bin = n_sites * pi * AR / NBIN * SPB_SCALE
NOISE_TARGET = 1200.0        # pseudo-particle weight cap, see mc_noise below
#   MC counting noise on theta is sqrt(weight / sites_per_bin); with the values
#   above it runs 2 % (AR 50) to 4 % (AR 10).  It is stored per example so that
#   training adds only the extra noise needed to reach the measurement level.
STOP_THETA = 0.999


# ---------------------------------------------------------------------- sampling
def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def draw_params(rng):
    p = dict(
        AR=_loguniform(rng, *PRIOR["AR"]),
        s0_ref=_loguniform(rng, *PRIOR["s0_ref"]),
        Ea=float(rng.uniform(*PRIOR["Ea"])),
        n_steric=float(rng.uniform(*PRIOR["n_steric"])),
        reemit=float(rng.uniform(*PRIOR["reemit"])),
        n_sites=_loguniform(rng, *PRIOR["n_sites"]),
    )
    return p


def s0_of_T(s0_ref, Ea, T):
    """Arrhenius in the sticking probability, referenced to T_REF, capped at 0.95."""
    s = s0_ref * np.exp(-(Ea / R_GAS) * (1.0 / T - 1.0 / T_REF))
    return np.clip(s, 1e-5, 0.95)


# ------------------------------------------------------------------- one example
def simulate_draw(p, seed):
    """Run the three temperatures and return (K, 3, NBIN) clean profiles."""
    AR = p["AR"]
    spb = p["n_sites"] * np.pi * AR / NBIN * SPB_SCALE
    weight = max(1.0, spb / NOISE_TARGET)
    ckpt = np.unique((PI2_CKPT * spb * NBIN / weight).astype(np.int64))
    budget = int(ckpt[-1])

    out = np.zeros((K_CKPT, 3, NBIN), dtype=np.float32)
    s0s = np.zeros(3)
    for j, T in enumerate(TEMPS_K):
        s0 = float(s0_of_T(p["s0_ref"], p["Ea"], T))
        s0s[j] = s0
        snaps, _, _ = run_ckpt(AR, s0, p["n_steric"], p["reemit"], spb,
                               weight, ckpt, budget, STOP_THETA, seed + j)
        out[:, j, :] = snaps.astype(np.float32)
    return out, spb, weight, s0s


def build_shard(shard_id):
    """Return the arrays for one shard. Seeds depend only on (shard_id, i)."""
    rng = np.random.default_rng(1_000_003 + shard_id)
    Y, C, P, Q = [], [], [], []
    for i in range(DRAWS_PER_SHARD):
        p = draw_params(rng)
        seed = shard_id * 100_003 + i * 3        # 3 consecutive seeds per draw
        prof, spb, weight, s0s = simulate_draw(p, seed)
        for k in range(K_CKPT):
            Y.append(prof[k])
            # known conditions: AR, dose per site, the three temperatures
            C.append([p["AR"], PI2_CKPT[k], *TEMPS_K])
            # inference targets (+ the realised s0(T) kept for diagnostics)
            P.append([p["s0_ref"], p["Ea"], p["n_steric"], p["reemit"],
                      p["n_sites"], *s0s])
            Q.append([spb, weight, np.sqrt(weight / spb)])
    return (np.asarray(Y, np.float32),
            np.asarray(C, np.float32),
            np.asarray(P, np.float32),
            np.asarray(Q, np.float32))


def run_shard(shard_id):
    path = os.path.join(OUT_DIR, f"shard_{shard_id:04d}.npz")
    if os.path.exists(path):
        return shard_id, 0.0, "skip"
    t0 = time.time()
    Y, C, P, Q = build_shard(shard_id)
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp, y=Y, c=C, p=P, q=Q,
        cond_names=np.array(["AR", "pi2_dose_per_site", "T1_K", "T2_K", "T3_K"]),
        param_names=np.array(["s0_ref", "Ea_kJmol", "n_steric", "reemit",
                              "n_sites", "s0_T1", "s0_T2", "s0_T3"]),
        diag_names=np.array(["sites_per_bin", "weight", "mc_noise"]),
        temps_K=TEMPS_K, T_ref_K=T_REF, pi2_ckpt=PI2_CKPT, nbin=NBIN)
    os.replace(tmp, path)                        # atomic: no half-written shard
    return shard_id, time.time() - t0, "done"


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=None,
                    help="subset, e.g. '0-7' or '3,9,11'")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = os.cpu_count() - 1")
    ap.add_argument("--draws", type=int, default=0,
                    help="override DRAWS_PER_SHARD (testing)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global DRAWS_PER_SHARD
    if args.draws:
        DRAWS_PER_SHARD = args.draws
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.shards is None:
        ids = list(range(N_SHARD))
    elif "-" in args.shards:
        a, b = args.shards.split("-")
        ids = list(range(int(a), int(b) + 1))
    else:
        ids = [int(x) for x in args.shards.split(",")]

    todo = [s for s in ids
            if not os.path.exists(os.path.join(OUT_DIR, f"shard_{s:04d}.npz"))]
    n_ex = len(todo) * DRAWS_PER_SHARD * K_CKPT
    print(f"output   : {OUT_DIR}")
    print(f"shards   : {len(ids)} requested, {len(todo)} missing")
    print(f"examples : {n_ex:,} to generate "
          f"({len(ids)*DRAWS_PER_SHARD*K_CKPT:,} in the full set)")
    print(f"estimate : {n_ex*15.0/3600:.1f} core-hours at ~15 s per example")
    if args.dry_run or not todo:
        return

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"workers  : {workers}\n")

    # Warm the numba JIT in the parent: with fork start-up the workers inherit
    # the compiled code instead of racing to compile it six times over.
    print("warming up numba ...", flush=True)
    run_ckpt(10.0, 0.05, 1.0, 1.0, 400.0, 1.0,
             np.array([50, 100], dtype=np.int64), 100, 0.999, 1)

    import multiprocessing as mp
    t0 = time.time()
    done = 0
    with mp.Pool(workers) as pool:
        for sid, dt, status in pool.imap_unordered(run_shard, todo):
            done += 1
            el = time.time() - t0
            eta = el / done * (len(todo) - done)
            print(f"  shard {sid:4d}  {status:5s}  {dt:7.1f}s   "
                  f"[{done}/{len(todo)}]  elapsed {el/60:.1f} min, "
                  f"eta {eta/60:.1f} min", flush=True)
    print(f"\nfinished in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
