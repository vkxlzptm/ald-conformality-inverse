"""Dataset loading, the observation model, and target transforms.

Everything the network sees and everything it predicts is dimensionless, so the
pipeline carries no dependence on the simulator's unit conventions.

    known    :  theta(z) normalised by the flat-top thickness,  AR = H/D
    inferred :  s0 (at the wafer's temperature), n, re-emission survival, Pi2
    outside  :  n_s   = N_dose / (4 AR Pi2)        with the measured dose
                Ea    = -k_B * slope of ln s0 vs 1/T   across a temperature split

Why Pi2 is an output and not an input: the profile depends on dose and site
count only through their ratio, and that ratio's denominator is the site density
we are trying to measure.  Feeding it in is circular -- measured in README 4-f.

Why the absolute dose is not an input at all: scaling dose and site count
together leaves the profile unchanged, so the dose carries no information the
profile does not already contain.

Split is by SHARD, never by example: the dose checkpoints and the three
temperatures of one draw share parameters, so an example-level split leaks.
"""
import glob
import os

import numpy as np

NBIN = 64
K_TEMP = 3
KB_EV = 8.617333262e-5          # eV / K
KJMOL_PER_EV = 96.48533212

TARGETS = ["s0", "n_steric", "reemit", "pi2"]
LOG_TARGETS = {"s0", "pi2"}

# Which dimensionless group the first target is expressed in.
#   "s0"  : the network predicts log s0 directly
#   "pi1" : it predicts log Pi1 = log AR + 0.5 log s0, the group the profile
#           shape actually collapses on (README 4-2)
# AR is a network input either way, so the two are in bijection and the network
# can construct one from the other.  The real difference is which quantity the
# loss is z-normalised in.  Which is better is measured, not assumed:
# src/ablation_param.py.  Everything outside this module works in physical units
# (s0), so nothing downstream has to know which one is in use.
TARGET_PARAM = "s0"
MEAS_NOISE = 0.03               # measurement noise the network is trained for
N_MASKED = 10                   # depth bins missing from a measurement


def shard_id(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def load(root, ids=None):
    """Load shards and expand each example into its three temperatures.

    One row of the returned arrays is one wafer: a single profile measured at a
    single temperature.
    """
    files = sorted(glob.glob(os.path.join(root, "shard_*.npz")))
    if ids is not None:
        keep = set(ids)
        files = [f for f in files if shard_id(f) in keep]
    if not files:
        raise FileNotFoundError(f"no shards under {root}")

    Y, AR, T, MC, P, EA, SH, GR = [], [], [], [], [], [], [], []
    for f in files:
        d = np.load(f)
        y, c, p, q = d["y"], d["c"], d["p"], d["q"]
        n, k = y.shape[0], len(d["pi2_ckpt"])
        if n % k:
            raise ValueError(f"{f}: {n} examples is not a multiple of {k}")
        sid = shard_id(f)
        temps = d["temps_K"]
        group = sid * 1_000_000 + np.arange(n)      # unique per (shard, example)
        for j in range(K_TEMP):
            Y.append(y[:, j, :])
            AR.append(c[:, 0])
            T.append(np.full(n, temps[j]))
            MC.append(q[:, 2])
            # target: s0 at THIS temperature, steric, re-emission, Pi2
            P.append(np.stack([p[:, 5 + j], p[:, 2], p[:, 3], c[:, 1]], 1))
            EA.append(p[:, 1] / KJMOL_PER_EV)       # truth, eV, for the Arrhenius check
            SH.append(np.full(n, sid, np.int32))
            GR.append(group)

    cat = lambda a: np.concatenate(a)
    return dict(y=cat(Y).astype(np.float32), AR=cat(AR).astype(np.float64),
                T=cat(T).astype(np.float64), mc_noise=cat(MC).astype(np.float64),
                p=cat(P).astype(np.float64), Ea_true=cat(EA).astype(np.float64),
                shard=cat(SH), group=cat(GR))


def split_ids(root, frac=(0.90, 0.05, 0.05)):
    ids = sorted(shard_id(f)
                 for f in glob.glob(os.path.join(root, "shard_*.npz")))
    n = len(ids)
    if n < 3:
        raise ValueError("need at least 3 shards to split by shard")
    a = min(max(1, int(round(frac[0] * n))), n - 2)
    b = min(max(a + 1, a + int(round(frac[1] * n))), n - 1)
    return ids[:a], ids[a:b], ids[b:]


# ------------------------------------------------------------ target transform
def set_target_param(mode):
    global TARGET_PARAM
    if mode not in ("s0", "pi1"):
        raise ValueError(f"target parametrisation must be s0 or pi1, got {mode}")
    TARGET_PARAM = mode


def _ar(AR):
    if AR is None:
        raise ValueError("AR is required when the first target is Pi1")
    return np.log(np.asarray(AR, dtype=np.float64))


def targets_to_z(p, AR=None, mode=None):
    """Physical targets -> the space the network is trained in (before z-scoring)."""
    mode = mode or TARGET_PARAM
    z = np.array(p, dtype=np.float64, copy=True)
    for j, name in enumerate(TARGETS):
        if name in LOG_TARGETS:
            z[:, j] = np.log(np.maximum(p[:, j], 1e-8))
    if mode == "pi1":
        z[:, 0] = _ar(AR) + 0.5 * z[:, 0]              # ln Pi1 = ln AR + ln sqrt(s0)
    return z


def z_to_targets(z, AR=None, mode=None):
    """Inverse of targets_to_z: always returns physical units, s0 included."""
    mode = mode or TARGET_PARAM
    out = np.array(z, dtype=np.float64, copy=True)
    if mode == "pi1":
        out[:, 0] = 2.0 * (out[:, 0] - _ar(AR))        # back to ln s0
    for j, name in enumerate(TARGETS):
        if name in LOG_TARGETS:
            out[:, j] = np.exp(out[:, j])
    return out


def ln_s0_from_z0(z0, AR=None, mode=None):
    """First target coordinate (un-z-scored) -> ln s0, in either parametrisation.

    Used by the Arrhenius fit, which needs ln s0 and nothing else.  Note that a
    constant offset would drop out of the slope anyway, since AR is fixed inside
    one temperature split -- but the factor of 2 does not, so this is not
    optional.
    """
    mode = mode or TARGET_PARAM
    return z0 if mode == "s0" else 2.0 * (z0 - _ar(AR))


# --------------------------------------------------------- observation model
def degrade(y, mc_noise, rng, meas_noise=MEAS_NOISE, n_masked=N_MASKED):
    """Apply the measurement model to clean profiles.

    The stored profiles already carry Monte Carlo counting noise of size
    `mc_noise`, so only the shortfall to the target measurement noise is added.
    Masking is redrawn every epoch, which doubles as augmentation.

    y : (N, NBIN);  returns obs (N, NBIN), mask (N, NBIN) with 1 = measured
    """
    n = y.shape[0]
    extra = np.sqrt(np.maximum(meas_noise ** 2 - np.asarray(mc_noise) ** 2, 0.0))
    obs = y * (1.0 + extra[:, None] * rng.standard_normal(y.shape))
    np.clip(obs, 0.0, 1.0, out=obs)

    mask = np.ones((n, NBIN), np.float32)
    idx = np.argsort(rng.random((n, NBIN)), axis=1)[:, :n_masked]
    np.put_along_axis(mask, idx, 0.0, axis=1)
    return obs.astype(np.float32), mask


def features(obs, mask, AR):
    """Network input.

    channel 0 : coverage, masked bins zeroed
    channel 1 : log10 of the same, so information deep in the feature -- where
                coverage is small -- is not squashed against zero
    channel 2 : the mask, so a missing bin is distinguishable from a bare one
    condition : log AR, the only scalar a process actually fixes that the
                profile does not already contain
    """
    obs = np.atleast_2d(obs).astype(np.float32)
    mask = np.atleast_2d(mask).astype(np.float32)
    lo = np.log10(np.maximum(obs, 1e-4))
    x = np.stack([obs * mask, (lo / 4.0 + 1.0) * mask, mask], axis=1)
    c = np.log(np.asarray(AR, dtype=np.float64))[:, None].astype(np.float32)
    return x.astype(np.float32), c


# ------------------------------------------------------------- unit recovery
def site_density(pi2, AR, dose_per_opening_area):
    """n_s = N_dose / (4 AR Pi2).

    The 4 AR is geometry, not a convention: the side wall of a cylindrical via
    is 4 AR times its opening.  The hole diameter cancels, so only the aspect
    ratio is needed -- no absolute dimension enters anywhere.
    """
    return np.asarray(dose_per_opening_area) / (4.0 * np.asarray(AR)
                                                * np.asarray(pi2))
