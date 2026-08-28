"""Can the known conditions alone give away a parameter?

The dose stored in the dataset was generated as

    dose = Pi2 * n_sites * pi * AR * SPB_SCALE ,   Pi2 on a fixed 6-point grid

so  log n_sites = log dose - log AR - const - log Pi2.  With Pi2 restricted to
six known values, (AR, dose) narrows the site density to a handful of discrete
candidates before the profile is even looked at.  A real process sets the dose
independently of the site density it is trying to measure, so any accuracy that
comes from this route is an artefact of how the training set was built.

This script measures the size of that shortcut two ways:
  1. enumerate the six Pi2 values and count how many give an n_sites inside the
     prior -- if it is usually one, the site density is simply given
  2. nearest neighbour on (log AR, log dose) alone, no profile at all
"""
import argparse
import os

import numpy as np

import data as D
import generate_dataset as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "results", "dataset"))
    a = ap.parse_args()

    tr_ids, _, te_ids = D.split_ids(a.data)
    tr = D.load(a.data, tr_ids[:60])
    te = D.load(a.data, te_ids)
    NB = D.NBIN
    const = np.pi * G.SPB_SCALE

    # ---- 1. how many of the six Pi2 values are consistent with (AR, dose)? ----
    lo, hi = G.PRIOR["n_sites"]
    cand = te["dose"][:, None] / (G.PI2_CKPT[None, :] * te["AR"][:, None] * const)
    inside = (cand >= lo) & (cand <= hi)
    n_cand = inside.sum(1)
    truth = te["p"][:, D.TARGETS.index("n_sites")]
    best = np.where(inside, np.abs(cand - truth[:, None]), np.inf).argmin(1)
    err_if_known = np.abs(cand[np.arange(len(cand)), best] - truth) / truth

    print("1. enumerating the six Pi2 values against (AR, dose)")
    for k in range(1, 7):
        f = (n_cand == k).mean()
        if f:
            print(f"     {k} candidate(s) inside the prior : {f*100:5.1f} % of cases")
    print(f"   picking the closest candidate reproduces n_sites to "
          f"{np.median(err_if_known)*100:.2f} % (median)")
    print("   -> the profile only has to choose between a few discrete options\n")

    # ---- 2. nearest neighbour on conditions only, no profile ------------------
    ftr = np.stack([np.log(tr["AR"]), np.log(tr["dose"])], 1)
    fte = np.stack([np.log(te["AR"]), np.log(te["dose"])], 1)
    mu, sd = ftr.mean(0), ftr.std(0)
    ftr = (ftr - mu) / sd
    fte = (fte - mu) / sd
    K = 8
    print(f"2. {K}-nearest-neighbour on (log AR, log dose) only, no profile")
    print("     parameter     median error from conditions alone")
    for j, name in enumerate(D.TARGETS):
        pred = np.empty(len(fte))
        for i in range(0, len(fte), 2000):
            d2 = ((fte[i:i + 2000, None, :] - ftr[None, :, :]) ** 2).sum(-1)
            idx = np.argpartition(d2, K, axis=1)[:, :K]
            pred[i:i + 2000] = np.median(tr["p"][idx, j], axis=1)
        t = te["p"][:, j]
        if name == "Ea":
            print(f"     {name:11s}   {np.median(np.abs(pred - t)):8.2f} kJ/mol")
        else:
            print(f"     {name:11s}   {np.median(np.abs(pred - t) / np.maximum(t, 1e-9))*100:8.2f} %")
    print("\n   A parameter recovered accurately here was never inferred from the\n"
          "   physics -- it was handed over by how the training set was built.")


if __name__ == "__main__":
    main()
