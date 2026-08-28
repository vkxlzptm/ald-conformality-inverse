"""Dataset loading, the observation model, and target transforms.

Split is by SHARD, never by example: the K dose checkpoints inside one parameter
draw share identical ground truth, so an example-level random split would put the
same answer in train and test and inflate every score.

The known process condition is the ABSOLUTE dose, not Pi2 -- see README section
4-f.  Shards store Pi2 and sites_per_bin, from which the absolute molecule count
is recovered as  pi2 * sites_per_bin * NBIN.
"""
import glob
import os

import numpy as np

NBIN = 64
TARGETS = ["s0_ref", "Ea", "n_steric", "reemit", "n_sites"]
LOG_TARGETS = {"s0_ref", "n_sites"}
MEAS_NOISE = 0.03          # measurement noise the network is trained to expect
N_MASKED = 10              # depth bins missing from a measurement


def shard_id(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def load(root, ids=None):
    """Load shards into memory. Returns dict of arrays."""
    files = sorted(glob.glob(os.path.join(root, "shard_*.npz")))
    if ids is not None:
        ids = set(ids)
        files = [f for f in files if shard_id(f) in ids]
    if not files:
        raise FileNotFoundError(f"no shards under {root}")
    Y, C, P, Q, S = [], [], [], [], []
    for f in files:
        d = np.load(f)
        n = d["y"].shape[0]
        Y.append(d["y"]); C.append(d["c"]); P.append(d["p"]); Q.append(d["q"])
        S.append(np.full(n, shard_id(f), np.int32))
    Y = np.concatenate(Y); C = np.concatenate(C)
    P = np.concatenate(P); Q = np.concatenate(Q); S = np.concatenate(S)

    pi2, spb = C[:, 1], Q[:, 0]
    dose = pi2 * spb * NBIN                       # absolute molecules entering
    return dict(y=Y, AR=C[:, 0], dose=dose, mc_noise=Q[:, 2],
                p=P[:, :5], shard=S)


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
def targets_to_z(p):
    """Physical parameters -> the space the network regresses in."""
    z = np.empty_like(p, dtype=np.float64)
    for j, name in enumerate(TARGETS):
        z[:, j] = np.log(np.maximum(p[:, j], 1e-8)) if name in LOG_TARGETS \
            else p[:, j]
    return z


def z_to_targets(z):
    out = np.array(z, dtype=np.float64, copy=True)
    for j, name in enumerate(TARGETS):
        if name in LOG_TARGETS:
            out[:, j] = np.exp(out[:, j])
    return out


# --------------------------------------------------------- observation model
def degrade(y, mc_noise, rng, meas_noise=MEAS_NOISE, n_masked=N_MASKED):
    """Apply the measurement model to clean profiles.

    The stored profiles already carry Monte Carlo counting noise of size
    `mc_noise`, so only the shortfall to the target measurement noise is added;
    otherwise the training data would be noisier than the stated instrument.
    Masking is redrawn every epoch, which doubles as augmentation.

    y : (N, 3, NBIN) float32
    returns obs (N, 3, NBIN), mask (N, NBIN) with 1 = measured
    """
    n = y.shape[0]
    extra = np.sqrt(np.maximum(meas_noise ** 2 - mc_noise ** 2, 0.0))
    obs = y * (1.0 + extra[:, None, None] * rng.standard_normal(y.shape))
    np.clip(obs, 0.0, 1.0, out=obs)

    mask = np.ones((n, NBIN), np.float32)
    idx = np.argsort(rng.random((n, NBIN)), axis=1)[:, :n_masked]
    np.put_along_axis(mask, idx, 0.0, axis=1)
    return obs.astype(np.float32), mask


def features(obs, mask, AR, dose):
    """Assemble the network input.

    channels 0-2  : coverage at the three temperatures, masked bins zeroed
    channels 3-5  : log10 of the same, so information deep in the feature (where
                    coverage is small) is not squashed against zero
    channel  6    : the mask itself, so the network can tell a missing bin from
                    a genuinely empty one
    conditions    : log AR and log dose, the two things a process actually knows
    """
    m = mask[:, None, :]
    lo = np.log10(np.maximum(obs, 1e-4))
    x = np.concatenate([obs * m, (lo / 4.0 + 1.0) * m, mask[:, None, :]],
                       axis=1).astype(np.float32)
    c = np.stack([np.log(AR), np.log(dose)], axis=1).astype(np.float32)
    return x, c
