"""Do the known conditions alone give a parameter away?

At inference the process knows the aspect ratio, the absolute dose it delivered
and the temperature.  The network is shown only the profile and log AR; the dose
enters afterwards, in the unit recovery

    n_s = N_dose / (4 AR Pi2)        with Pi2 predicted from the profile,

and that line is where a badly built training set can cheat.  The generator draws

    N_dose = Pi2 * n_sites * pi * AR * SPB_SCALE ,

so while Pi2 sat on the fixed six-point checkpoint grid, dividing the known dose
by those six values returned the site density before the profile was read at all
-- measured at 0.00 % error.  A random log offset per draw (PI2_JITTER) smears
the grid into a continuum to close that route.  This script is the gate on
whether the smearing is really present in the shards on disk.

Every test here is profile-free by construction: accuracy reported below was
handed over by the conditions, not inferred from the physics.

  1. grid enumeration  divide the known dose by the six nominal Pi2 values and
                       see how well the best candidate reproduces n_sites
  2. support width     how much of the n_sites prior survives once (AR, dose)
                       and the full support of Pi2 are accounted for
  3. conditions-only   k-nearest neighbour on the knowns alone, against a
     kNN               predict-the-prior-median null

Tests 1 and 3 are also reported with the jitter removed analytically (Pi2 snapped
back to the grid, dose recomputed).  That column is the leaky v1 dataset and is
there to show the tests can still see a leak when one exists -- a gate that
cannot fail proves nothing.

Only the columns the network can actually reach are gated.  kNN on (AR, dose) is
printed for context but not gated: the network never sees the dose, and knowing
(AR, dose) pins the product Pi2 * n_sites, so any apparent skill on Pi2 there is
borrowed from the narrow n_sites prior rather than from the physics.  That prior
correlation is a property of the sampling design, stated in README 4-f.

    python src/leak_test.py
    python src/leak_test.py --data results/dataset_v1_leaky

The shards are read directly rather than through data.py: a leak test that shares
the training pipeline's assumptions cannot check them.  Columns are resolved by
the name arrays stored in each shard, not by position.
"""
import argparse
import glob
import os

import numpy as np

import data as D
import generate_dataset as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNN_K = 8
PARAMS = ["s0", "n_steric", "reemit", "pi2", "n_sites"]
ABS_ERROR = {"reemit"}          # sits near 1, a relative error would flatter it

GATE_GRID_ERR = 0.05            # test 1: best grid candidate must miss by >5 %
GATE_PRIOR_KEPT = 0.90          # test 2: >90 % of the n_sites prior must survive
GATE_KNN_RATIO = 0.50           # test 3: kNN(AR) may not halve the null error


# --------------------------------------------------------------------- loading
def load_conditions(root, ids, max_rows=0, seed=0):
    """One row per wafer: the knowns and the truths, no profiles."""
    keep = set(ids)
    files = [f for f in sorted(glob.glob(os.path.join(root, "shard_*.npz")))
             if int(os.path.basename(f).split("_")[1].split(".")[0]) in keep]
    if not files:
        raise FileNotFoundError(f"no shards under {root}")

    cols = {k: [] for k in ["AR", "pi2", "spb", "n_sites", "n_steric",
                            "reemit", "s0"]}
    nbin = None
    for f in files:
        d = np.load(f)
        cn = [str(x) for x in d["cond_names"]]
        pn = [str(x) for x in d["param_names"]]
        qn = [str(x) for x in d["diag_names"]]
        c, p, q = d["c"], d["p"], d["q"]
        nbin = int(d["nbin"])
        base = dict(
            AR=c[:, cn.index("AR")].astype(np.float64),
            pi2=c[:, cn.index("pi2_dose_per_site")].astype(np.float64),
            spb=q[:, qn.index("sites_per_bin")].astype(np.float64),
            n_sites=p[:, pn.index("n_sites")].astype(np.float64),
            n_steric=p[:, pn.index("n_steric")].astype(np.float64),
            reemit=p[:, pn.index("reemit")].astype(np.float64))
        for j in range(len(d["temps_K"])):          # one row per wafer
            for k, v in base.items():
                cols[k].append(v)
            cols["s0"].append(p[:, pn.index(f"s0_T{j + 1}")].astype(np.float64))

    w = {k: np.concatenate(v) for k, v in cols.items()}
    w["nbin"] = nbin
    if max_rows and len(w["AR"]) > max_rows:
        idx = np.random.default_rng(seed).choice(len(w["AR"]), max_rows,
                                                 replace=False)
        w = {k: (v[idx] if k != "nbin" else v) for k, v in w.items()}
    return w


def dose_of(w, snap_to_grid=False):
    """Absolute molecules into the opening: dose = Pi2 * (total sites)."""
    pi2 = w["pi2"]
    if snap_to_grid:
        j = np.abs(np.log(pi2)[:, None]
                   - np.log(G.PI2_CKPT)[None, :]).argmin(1)
        pi2 = G.PI2_CKPT[j]
    return pi2 * w["spb"] * w["nbin"], pi2


def n_sites_from(dose, AR, pi2):
    """Invert  dose = Pi2 * n_sites * pi * AR * SPB_SCALE."""
    return dose / (pi2 * np.pi * AR * G.SPB_SCALE)


# ----------------------------------------------------------------------- tests
def test_grid(w, dose):
    """Enumerate the six nominal Pi2 values against the known (AR, dose)."""
    lo, hi = G.PRIOR["n_sites"]
    cand = n_sites_from(dose[:, None], w["AR"][:, None], G.PI2_CKPT[None, :])
    inside = (cand >= lo) & (cand <= hi)
    truth = w["n_sites"]
    pick = np.where(inside, np.abs(cand - truth[:, None]), np.inf).argmin(1)
    err = np.abs(cand[np.arange(len(cand)), pick] - truth) / truth
    err = np.where(inside.any(1), err, np.inf)      # no candidate = no shortcut
    return inside.sum(1), float(np.median(err))


def test_support(w, dose):
    """Fraction of the log n_sites prior still allowed by (AR, dose)."""
    lo, hi = np.log(G.PRIOR["n_sites"])
    pi2_lo = G.PI2_CKPT[0] * np.exp(-G.PI2_JITTER)
    pi2_hi = G.PI2_CKPT[-1] * np.exp(G.PI2_JITTER)
    a = np.log(n_sites_from(dose, w["AR"], pi2_hi))     # smallest n_sites
    b = np.log(n_sites_from(dose, w["AR"], pi2_lo))     # largest
    kept = np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0.0, None) / (hi - lo)
    return kept, pi2_hi / pi2_lo


def knn(ftr, ytr, fte, k=KNN_K, chunk=1024):
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-12
    a, b = (ftr - mu) / sd, (fte - mu) / sd
    out = np.empty((len(b), ytr.shape[1]))
    for i in range(0, len(b), chunk):
        d2 = ((b[i:i + chunk, None, :] - a[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        out[i:i + chunk] = np.median(ytr[idx], axis=1)
    return out


def med_err(pred, truth):
    """Median error per parameter: absolute for re-emission, per cent otherwise."""
    e = []
    for j, name in enumerate(PARAMS):
        d = np.abs(pred[:, j] - truth[:, j])
        e.append(float(np.median(d)) if name in ABS_ERROR
                 else float(np.median(d / np.maximum(np.abs(truth[:, j]),
                                                     1e-12)) * 100))
    return np.array(e)


def fmt(vals):
    return "  ".join(f"{v:9.2f}" for v in vals)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "results", "dataset"))
    ap.add_argument("--train-rows", type=int, default=40000)
    ap.add_argument("--test-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr_ids, _, te_ids = D.split_ids(a.data)
    tr = load_conditions(a.data, tr_ids, a.train_rows, a.seed)
    te = load_conditions(a.data, te_ids, a.test_rows, a.seed + 1)
    print(f"data     : {a.data}")
    print(f"rows     : {len(tr['AR']):,} train ({len(tr_ids)} shards), "
          f"{len(te['AR']):,} test ({len(te_ids)} shards)")
    print(f"jitter   : PI2_JITTER = {G.PI2_JITTER} "
          f"(half a grid spacing is {np.log(G.PI2_CKPT[-1] / G.PI2_CKPT[0]) / (len(G.PI2_CKPT) - 1) / 2:.4f})\n")

    dose_tr, _ = dose_of(tr)
    dose_te, pi2_te = dose_of(te)
    dose_tr_ng, _ = dose_of(tr, snap_to_grid=True)
    dose_te_ng, _ = dose_of(te, snap_to_grid=True)

    # consistency of the dose formula against the stored truth
    chk = np.abs(n_sites_from(dose_te, te["AR"], pi2_te) / te["n_sites"] - 1)
    print(f"sanity   : dose formula reproduces n_sites to "
          f"{chk.max():.2e} (max relative)\n")

    # how far the stored Pi2 sits from the nominal grid, in half-spacings
    off = np.log(te["pi2"])[:, None] - np.log(G.PI2_CKPT)[None, :]
    off = off[np.arange(len(off)), np.abs(off).argmin(1)] / G.PI2_JITTER
    print("0. is the checkpoint grid actually smeared out?")
    print(f"     |offset| from the nearest nominal Pi2, in units of PI2_JITTER:")
    print(f"     median {np.median(np.abs(off)):.3f}, "
          f"90th pct {np.quantile(np.abs(off), 0.90):.3f}   "
          f"(uniform smearing gives 0.50 and 0.90)")
    print(f"     fraction within 1 % of a nominal value: "
          f"{(np.abs(off) * G.PI2_JITTER < 0.01).mean() * 100:.2f} % "
          f"(a fixed grid gives 100 %)\n")

    # ---- 1. enumerate the six nominal Pi2 values ----------------------------
    n_cand, err = test_grid(te, dose_te)
    n_cand_ng, err_ng = test_grid(te, dose_te_ng)
    print("1. enumerating the six nominal Pi2 values against (AR, dose)")
    print("     candidates inside the n_sites prior       as generated   "
          "jitter removed")
    for k in range(0, len(G.PI2_CKPT) + 1):
        f, g = (n_cand == k).mean(), (n_cand_ng == k).mean()
        if f or g:
            print(f"       {k} candidate(s)                          "
                  f"{f * 100:7.1f} %      {g * 100:7.1f} %")
    print(f"     best candidate reproduces n_sites to    "
          f"{err * 100:8.2f} %      {err_ng * 100:7.2f} %")
    print("     -> a small number here means the site density was handed over\n")

    # ---- 2. how much of the n_sites prior survives --------------------------
    kept, span = test_support(te, dose_te)
    prior_span = G.PRIOR["n_sites"][1] / G.PRIOR["n_sites"][0]
    print("2. width of the Pi2 support against the width of the n_sites prior")
    print(f"     Pi2 support spans {span:.1f} x, the n_sites prior spans "
          f"{prior_span:.1f} x")
    print(f"     fraction of the log n_sites prior still allowed by (AR, dose):")
    print(f"       median {np.median(kept):.3f},  "
          f"10th pct {np.quantile(kept, 0.10):.3f},  "
          f"fully intact in {(kept > 0.999).mean() * 100:.1f} % of wafers\n")

    # ---- 3. nearest neighbour on the conditions only ------------------------
    ytr = np.stack([tr[k] for k in PARAMS], 1)
    yte = np.stack([te[k] for k in PARAMS], 1)
    null = med_err(np.tile(np.median(ytr, 0), (len(yte), 1)), yte)

    f1_tr = np.log(tr["AR"])[:, None]
    f1_te = np.log(te["AR"])[:, None]
    e_ar = med_err(knn(f1_tr, ytr, f1_te), yte)

    f2 = lambda w, d: np.stack([np.log(w["AR"]), np.log(d)], 1)
    e_ard = med_err(knn(f2(tr, dose_tr), ytr, f2(te, dose_te)), yte)
    e_ard_ng = med_err(knn(f2(tr, dose_tr_ng), ytr, f2(te, dose_te_ng)), yte)

    print(f"3. {KNN_K}-nearest neighbour on the knowns alone, no profile")
    print("   median error (%, re-emission in absolute units)")
    print("                        " + "  ".join(f"{n:>9s}" for n in PARAMS))
    print("     prior median null  " + fmt(null))
    print("     kNN on AR          " + fmt(e_ar) + "     <- what the network sees")
    print("     kNN on AR, dose    " + fmt(e_ard) + "     (context, not gated)")
    print("     same, jitter gone  " + fmt(e_ard_ng) + "     <- the v1 leak")
    print("     ratio to null (AR) " + fmt(e_ar / np.maximum(null, 1e-12)))
    print()

    # ------------------------------------------------------------------ gate
    g1 = err > GATE_GRID_ERR
    g2 = float(np.median(kept)) > GATE_PRIOR_KEPT
    worst = float(np.min(e_ar / np.maximum(null, 1e-12)))
    g3 = worst > GATE_KNN_RATIO
    print("gate")
    print(f"  1  grid candidate misses n_sites by > {GATE_GRID_ERR * 100:.0f} % "
          f"      : {err * 100:6.2f} %   {'pass' if g1 else 'FAIL'}")
    print(f"  2  median of the n_sites prior kept > {GATE_PRIOR_KEPT:.2f}"
          f"      : {np.median(kept):6.3f}     {'pass' if g2 else 'FAIL'}")
    print(f"  3  kNN(AR) / null error > {GATE_KNN_RATIO:.2f} for every target"
          f": {worst:6.3f}     {'pass' if g3 else 'FAIL'}")
    ok = g1 and g2 and g3
    print(f"\n  {'PASS -- no parameter is reachable from the conditions alone.' if ok else 'FAIL -- the conditions still give a parameter away; do not train on this set.'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
