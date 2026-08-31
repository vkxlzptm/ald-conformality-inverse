"""Validation 3: does the inference carry to an aspect ratio it never measured?

This is the question a process engineer actually asks.  A wafer is measured at one
aspect ratio with one dose.  How much dose does the *next* structure need?

    measure at AR_src  ->  posterior over (s0, n, re-emission, Pi2)
                       ->  simulate each posterior sample at AR_tgt
                       ->  a band on the dose the deeper structure needs
                       vs  the same quantity from the true parameters

The answer is reported as a multiplier on the dose already used, which makes it
free of every unit in the pipeline.  With the same areal site density at both
aspect ratios the total site count scales as AR, so

    N_dose(tgt)     Pi2(tgt)     AR_tgt
    ----------- =  ---------- *  ------
    N_dose(src)     Pi2(src)     AR_src

and both Pi2 come from the same posterior sample, so their errors partly cancel.
The site density, SPB_SCALE and the absolute hole diameter never appear.

Only the *transport* is re-simulated at the new aspect ratio; s0, the steric
exponent and the re-emission survival are surface properties and carry over
unchanged.  That is the assumption this test is checking, and it is the reason a
parameter inference is worth more than a curve fit to step coverage.

Which cases.  The training prior lets the per-bounce loss (1 - re-emission)
exceed s0, and in that regime a precursor molecule is destroyed before it reacts
anywhere: measured, 44 % of prior draws have loss > s0 and no dose fills a deep
feature.  That is outside any usable ALD window, so a transfer study there
measures the prior, not the method.  Cases are therefore drawn from the
ALD-like part of the prior, s0 > MIN_RATIO * (1 - re-emission), and the filter
is printed with the results rather than hidden.  The network itself is trained
on the full prior, which is the conservative direction.

What is reported.  The headline is the whole dose -> bottom coverage curve at the
new aspect ratio, band against truth: it is always defined.  The dose multiplier
is derived from it for a coverage target, by default the coverage the source
wafer already achieved, i.e. "hold this conformality at a deeper feature".  Where
the target is out of reach inside the dose grid the case is reported as censored
rather than silently clipped.

Cost.  One MC run per posterior sample, run out along the dose axis and
snapshotted, so a whole dose curve costs one run (src/cyl_run.py).  The particle
budget of a run is about  Pi2_max * NBIN / noise^2, independent of the site
count.  Use --dry-run for the estimate and a measured per-run time before
committing to a large sample count.

    python src/transfer_ar.py --dry-run
    python src/transfer_ar.py --cases 6 --samples 100 --workers 6
    python src/transfer_ar.py --cases 6 --samples 100 --ls-budget 1500   # + baseline

The least-squares comparison is off by default: it costs another ~1500 simulator
calls per case, which is the whole point of the amortized network.
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
import generate_dataset as G
import baseline_ls as B
from cyl_mc import NBIN
from cyl_run import run_ckpt

# torch is imported inside main() so that --dry-run, and the simulator helpers
# below, work on a machine that only has to generate data.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPB = B.SPB_REF                 # numerical resolution only, cancels from the physics


# ------------------------------------------------------- required dose per site
def pi2_grid(lo, hi, n):
    return np.geomspace(lo, hi, n)


def checkpoints(grid, weight):
    """Pseudo-particle counts, and the Pi2 they actually correspond to.

    Rounding to whole particles and dropping duplicates can shorten the grid, so
    the realised Pi2 axis is derived here rather than assumed to be `grid`.
    """
    ck = np.unique(np.maximum((grid * SPB * NBIN / weight).astype(np.int64), 1))
    return ck, ck * weight / (SPB * NBIN)


def required_pi2(s0, n_steric, reemit, AR, grid, weight, target, seed):
    """Dose per site at which the deepest bin first reaches `target` coverage.

    Returns (pi2_required, bottom_coverage_curve).  np.inf means the target was
    not reached inside the grid -- censored, and reported as such rather than
    quietly clipped.
    """
    ck, pi2 = checkpoints(grid, weight)
    snaps, _, _ = run_ckpt(AR, float(s0), float(n_steric), float(reemit),
                           SPB, weight, ck, int(ck[-1]),
                           min(0.9995, target + 0.005), int(seed))
    bot = np.maximum.accumulate(snaps[:, NBIN - 1])     # monotone by construction
    j = int(np.searchsorted(bot, target))
    if j == 0:
        return float(pi2[0]), bot, pi2
    if j >= len(bot):
        return np.inf, bot, pi2
    f = (target - bot[j - 1]) / max(bot[j] - bot[j - 1], 1e-12)
    return float(np.exp(np.log(pi2[j - 1])
                        + f * (np.log(pi2[j]) - np.log(pi2[j - 1])))), bot, pi2


_JOB = {}


def _init(job):
    _JOB.update(job)
    run_ckpt(10.0, 0.05, 1.0, 1.0, 400.0, 1.0,
             np.array([50], dtype=np.int64), 50, 0.999, 1)


def _one(task):
    """task = (case, sample index, s0, n_steric, reemit, target)."""
    ci, si, s0, n_st, re, target = task
    t0 = time.time()
    q, bot, _ = required_pi2(s0, n_st, re, _JOB["AR_tgt"], _JOB["grid"],
                             _JOB["weight"], target,
                             seed=90_000 + ci * 10_007 + si)
    return ci, si, q, bot, time.time() - t0


def pick_cases(n, AR_src, min_ratio, scan=400):
    """Benchmark cases from the ALD-like part of the prior.

    A draw with s0 <= min_ratio * (1 - reemit) loses precursor faster than it
    consumes it, so no dose coats a deep feature and the transfer question has
    no answer to check.  Indices come from the same reproducible stream as the
    least-squares benchmark, so a case kept here is the same case there.
    """
    keep = []
    for i in range(scan):
        t, o, m, AR, T = B.make_case(i, AR=AR_src)
        if t["s0"] > min_ratio * (1.0 - t["reemit"]):
            keep.append((i, t, o, m, AR, T))
            if len(keep) == n:
                break
    if len(keep) < n:
        raise SystemExit(f"only {len(keep)} of {scan} draws pass the filter; "
                         f"lower --min-ratio or raise the scan")
    return keep


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "results", "model",
                                                    "best.pt"))
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--ar-src", type=float, default=20.0)
    ap.add_argument("--ar-tgt", type=float, default=50.0)
    ap.add_argument("--target", type=float, default=0.0,
                    help="bottom coverage the recipe has to reach; 0 means "
                         "match what the source wafer achieved")
    ap.add_argument("--min-ratio", type=float, default=3.0,
                    help="keep cases with s0 > min_ratio * (1 - reemit); the "
                         "ALD-like part of the prior, see the module docstring")
    ap.add_argument("--pi2-max", type=float, default=600.0,
                    help="top of the dose grid; the required dose at a\n                          higher AR runs well past the training prior")
    ap.add_argument("--pi2-min", type=float, default=0.5)
    ap.add_argument("--grid", type=int, default=60)
    ap.add_argument("--noise", type=float, default=0.05,
                    help="MC counting noise on theta in the transfer runs")
    ap.add_argument("--ls-budget", type=int, default=0,
                    help="simulator calls for the least-squares comparison "
                         "(0 = skip it)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", default=os.path.join(ROOT, "results",
                                                   "transfer_ar.json"))
    a = ap.parse_args()

    grid = pi2_grid(a.pi2_min, a.pi2_max, a.grid)
    weight = max(1.0, a.noise ** 2 * SPB)
    per_run = a.pi2_max * SPB * NBIN / weight
    n_runs = a.cases * (a.samples + 1)
    print(f"transfer  AR {a.ar_src:.0f} -> {a.ar_tgt:.0f}")
    print(f"target    " + (f"bottom coverage {a.target:.2f}" if a.target > 0
                           else "the bottom coverage the source wafer achieved"))
    print(f"cases     s0 > {a.min_ratio:g} x (1 - reemit): the ALD-like part of "
          f"the prior, see the module docstring")
    print(f"runs      {n_runs} ({a.cases} cases x {a.samples} posterior samples "
          f"+ 1 truth each)")
    print(f"budget    <= {per_run:,.0f} pseudo-particles per run "
          f"(theta noise {np.sqrt(weight / SPB) * 100:.1f} %), "
          f"{n_runs * per_run / 1e6:,.0f} M in total")
    print("          adaptive stop cuts this a lot whenever the target is "
          "reached early\n")
    if a.dry_run:
        return

    # ---------------------------------------------------------- the posterior
    import torch
    from evaluate import load_model, posterior
    dev = torch.device(a.device)
    net, zmu, zsd = load_model(a.model, dev)
    picked = pick_cases(a.cases, a.ar_src, a.min_ratio)
    idxs = [k[0] for k in picked]
    xs, cs, truths, obss, masks, targets = [], [], [], [], [], []
    print(f"\n  case   index      s0   reemit   s0/(1-reemit)   "
          f"source bottom coverage")
    for k, (i, t, o, m, AR, T) in enumerate(picked):
        xi, ci = D.features(o[None].astype(np.float32),
                            m[None].astype(np.float32), np.array([AR]))
        xs.append(xi); cs.append(ci)
        truths.append(t); obss.append(o); masks.append(m)
        # The spec is read off the source wafer.  The clean profile is used, so
        # the 3 % measurement noise on the spec itself is not propagated -- it is
        # a number the engineer chooses, not a quantity being inferred.
        src = B.forward(t, a.ar_src, seed=B.BENCH_SHARD * 100_003 + i)
        tgt = float(a.target) if a.target > 0 else float(src[NBIN - 1])
        targets.append(tgt)
        print(f"  {k:4d}   {i:5d}   {t['s0']:6.4f}   {t['reemit']:6.4f}   "
              f"{t['s0']/(1-t['reemit']):13.1f}   {src[NBIN-1]:10.3f}"
              f"{'   <- target' if a.target <= 0 else ''}")
    samp, _ = posterior(net, np.concatenate(xs), np.concatenate(cs), dev,
                        n_samp=a.samples)
    post = np.stack([D.z_to_targets(samp[i] * zsd + zmu, a.ar_src)
                     for i in range(a.cases)])          # (cases, samples, 4)
    jj = {n: D.TARGETS.index(n) for n in D.TARGETS}

    # ------------------------------------------------- least squares, optional
    ls = [None] * a.cases
    if a.ls_budget:
        print(f"least squares: {a.ls_budget} calls per case ...", flush=True)
        for i in range(a.cases):
            est, f, n, dt = B.fit(obss[i], masks[i].astype(bool), a.ar_src,
                                  a.ls_budget, 2, "Nelder-Mead")
            ls[i] = est
            print(f"  case {i}: {n} calls, {dt/60:.1f} min, sse {f:.4f}",
                  flush=True)

    # --------------------------------------------------------- transfer runs
    tasks = []
    for i in range(a.cases):
        tg = targets[i]
        for s in range(a.samples):
            tasks.append((i, s, post[i, s, jj["s0"]], post[i, s, jj["n_steric"]],
                          post[i, s, jj["reemit"]], tg))
        tasks.append((i, -1, truths[i]["s0"], truths[i]["n_steric"],
                      truths[i]["reemit"], tg))        # truth
        if ls[i] is not None:
            tasks.append((i, -2, ls[i]["s0"], ls[i]["n_steric"],
                          ls[i]["reemit"], tg))        # least squares

    job = dict(AR_tgt=a.ar_tgt, grid=grid, weight=weight)
    workers = a.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"\nsimulating at AR {a.ar_tgt:.0f} on {workers} workers "
          f"({len(tasks)} runs) ...", flush=True)

    res = {}
    curves = {}
    t0 = time.time()
    if workers == 1:
        _init(job)
        out = map(_one, tasks)
    else:
        import multiprocessing as mp
        pool = mp.Pool(workers, initializer=_init, initargs=(job,))
        out = pool.imap_unordered(_one, tasks)
    done = 0
    for ci, si, q, bot, dt in out:
        res[(ci, si)] = q
        curves[(ci, si)] = bot
        done += 1
        if done % max(1, len(tasks) // 10) == 0:
            el = time.time() - t0
            print(f"  {done}/{len(tasks)}   elapsed {el/60:.1f} min, "
                  f"eta {el/done*(len(tasks)-done)/60:.1f} min", flush=True)
    if workers != 1:
        pool.close(); pool.join()

    # ------------------------------------------------------------- the answer
    scale = a.ar_tgt / a.ar_src
    recs = []
    print(f"\n  dose multiplier to hold the source wafer's bottom coverage "
          f"at AR {a.ar_tgt:.0f},")
    print(f"  relative to the dose already used at AR {a.ar_src:.0f}\n")
    print("  case      truth      posterior median   68 % band          "
          "90 % band        in band   censored")
    inside68 = inside90 = 0
    for i in range(a.cases):
        q_true = res[(i, -1)]
        m_true = q_true / truths[i]["pi2"] * scale
        q = np.array([res[(i, s)] for s in range(a.samples)])
        p2 = post[i, :, jj["pi2"]]
        ok = np.isfinite(q)
        m = q[ok] / p2[ok] * scale
        cens = 1.0 - ok.mean()
        if ok.mean() < 0.80 or not np.isfinite(m_true):
            print(f"  {i:3d}   censored: the target is out of reach inside "
                  f"Pi2 <= {a.pi2_max:g} for {(1-ok.mean())*100:.0f} % of the "
                  f"posterior" + ("" if np.isfinite(m_true) else " and for the truth"))
            continue
        lo68, hi68 = np.quantile(m, [0.16, 0.84])
        lo90, hi90 = np.quantile(m, [0.05, 0.95])
        in68 = lo68 <= m_true <= hi68
        in90 = lo90 <= m_true <= hi90
        inside68 += in68; inside90 += in90
        rec = dict(case=i, truth=float(m_true), median=float(np.median(m)),
                   q16=float(lo68), q84=float(hi68), q05=float(lo90),
                   q95=float(hi90), censored=float(cens),
                   AR_src=a.ar_src, AR_tgt=a.ar_tgt, draw=int(idxs[i]),
                   target=float(targets[i]),
                   s0_true=float(truths[i]["s0"]),
                   reemit_true=float(truths[i]["reemit"]),
                   pi1_src=float(a.ar_src * np.sqrt(truths[i]["s0"])))
        if ls[i] is not None and np.isfinite(res[(i, -2)]):
            rec["ls"] = float(res[(i, -2)] / ls[i]["pi2"] * scale)
        recs.append(rec)
        extra = f"   LS {rec['ls']:6.2f}" if "ls" in rec else ""
        print(f"  {i:3d}   {m_true:8.2f}   {np.median(m):12.2f}      "
              f"{lo68:5.2f} - {hi68:5.2f}     {lo90:5.2f} - {hi90:5.2f}"
              f"     {'yes' if in90 else 'NO ':3s}     {cens*100:4.0f} %{extra}")

    n = len(recs)
    if n:
        print(f"\n  truth inside the 68 % band: {inside68}/{n}     "
              f"inside the 90 % band: {inside90}/{n}")
        rel = np.array([abs(r["median"] - r["truth"]) / r["truth"] for r in recs])
        print(f"  median error of the posterior median: {np.median(rel)*100:.1f} %")
        if any("ls" in r for r in recs):
            lrel = np.array([abs(r["ls"] - r["truth"]) / r["truth"]
                             for r in recs if "ls" in r])
            print(f"  least squares, point estimate       : "
                  f"{np.median(lrel)*100:.1f} %  (no band at all)")
    print(f"\n  wall clock {(time.time()-t0)/60:.1f} min on {workers} workers")

    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as fh:
        json.dump(dict(config=vars(a), cases=recs), fh, indent=1,
                  default=lambda o: None)
    print(f"  wrote {a.json}")

    # ------------------------------------------------------------------ figure
    make_figure(recs, curves, checkpoints(grid, weight)[1], a)


def make_figure(recs, curves, pi2_axis, a):
    if not recs:
        return
    show = recs[0]["case"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    ax = axes[0]
    band = np.stack([curves[(show, s)] for s in range(a.samples)])
    lo, mid, hi = np.quantile(band, [0.05, 0.5, 0.95], axis=0)
    ax.fill_between(pi2_axis, lo, hi, color="#2874a6", alpha=0.22,
                    label="Posterior, 90 % band")
    ax.plot(pi2_axis, mid, color="#2874a6", lw=2, label="Posterior median")
    ax.plot(pi2_axis, curves[(show, -1)], "k--", lw=2, label="Truth")
    ax.axhline(recs[0]["target"], color="#c0392b", lw=1.2, ls=":",
               label=f"Target coverage {recs[0]['target']:.2f}"
                     " (held from the source wafer)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Dose per surface site $\Pi_2$ at the new aspect ratio",
                  fontsize=12)
    ax.set_ylabel("Coverage at the bottom of the feature", fontsize=12)
    ax.set_title(f"Case {show}: dose curve predicted at AR {a.ar_tgt:.0f}\n"
                 f"from a single measurement at AR {a.ar_src:.0f}", fontsize=12)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9, loc="lower right")

    ax = axes[1]
    x = np.arange(len(recs))
    med = np.array([r["median"] for r in recs])
    ax.errorbar(x, med,
                yerr=[med - [r["q05"] for r in recs],
                      [r["q95"] for r in recs] - med],
                fmt="o", ms=6, lw=1.6, capsize=4, color="#2874a6",
                label="Amortized posterior, 90 % band")
    ax.plot(x, [r["truth"] for r in recs], "k_", ms=22, mew=2.4, label="Truth")
    if any("ls" in r for r in recs):
        xs = [i for i, r in enumerate(recs) if "ls" in r]
        ax.plot(xs, [recs[i]["ls"] for i in xs], "x", ms=9, mew=2.2,
                color="#c0392b", label="Least squares, point estimate")
    ax.set_xticks(x); ax.set_xticklabels([str(r["draw"]) for r in recs])

    ax.set_xlabel("Benchmark case (prior draw index)", fontsize=12)
    ax.set_ylabel(f"Dose multiplier for AR {a.ar_src:.0f} "
                  fr"$\rightarrow$ {a.ar_tgt:.0f}", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Required dose at an aspect ratio never measured", fontsize=12)
    ax.grid(alpha=0.3, axis="y", which="both"); ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "ar_transfer.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
