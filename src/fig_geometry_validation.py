"""Validation figure for the geometry decision.

(a) Zero-reaction transmission probability: 3D cylinder MC against a deterministic
    solution of the Clausing integral equation, and 2D slit MC against ln(AR)/AR.
(b) Profile-shape collapse in the 3D cylinder: AR*s0 versus AR*sqrt(s0).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cyl_mc as C3
from clausing_ref import clausing
from ar_scaling import transmission as trans_slit

NB = C3.NBIN
zc = (np.arange(NB) + 0.5) / NB


def shape(AR, s0, dose_mult=0.15, seed=7):
    spb = 400.0 * AR
    th, _ = C3.profile(AR, int(dose_mult * spb * NB), s0, sites_per_bin=spb, seed=seed)
    th = th[:-1]
    return th / max(th[0], 1e-12)


fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.3))
fig.subplots_adjust(wspace=0.30)

# ---- (a) transmission -------------------------------------------------
ax = axes[0]
ARs = np.array([0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0])
mc3 = np.array([C3.transmission(a, 400_000, 5)[0] for a in ARs])
ARc = np.geomspace(0.4, 50, 40)
det = np.array([clausing(a, M=800) for a in ARc])
ARs2 = np.array([2.0, 5.0, 10.0, 20.0, 40.0])
mc2 = np.array([trans_slit(a, 200_000, 5) for a in ARs2])

ax.plot(ARc, det, "-", lw=2.2, color="#2874a6",
        label="Clausing equation, deterministic")
ax.plot(ARs, mc3, "o", ms=8, mfc="none", mew=2, color="#154360",
        label="3D cylinder, ballistic MC")
ax.plot(ARs2, mc2, "s", ms=7, color="#c0392b", label="2D slit, ballistic MC")
ax.plot(ARc, np.log(ARc) / ARc, "--", lw=1.6, color="#c0392b", alpha=0.7,
        label=r"2D asymptote  $\ln(AR)/AR$")
ax.plot(ARc, 4 / (3 * ARc), ":", lw=1.6, color="#2874a6",
        label=r"Knudsen asymptote  $4/(3\,AR)$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Aspect ratio  AR = H / D", fontsize=12)
ax.set_ylabel("Transmission probability at $s_0$ = 0", fontsize=12)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9.5, loc="lower left")
ax.set_title("(a) Transport check against an exact reference\n"
             "cylinder MC matches the Clausing solution to < 1 %", fontsize=12.5)

# ---- (b) residual -----------------------------------------------------
ax = axes[1]
detp = np.array([clausing(a, M=1200) for a in ARs])
ax.axhline(0, color="k", lw=1)
ax.plot(ARs, 100 * (mc3 / detp - 1), "o-", ms=8, lw=1.8, color="#154360")
band = 100 * np.sqrt(detp * (1 - detp) / 400_000) / detp
ax.fill_between(ARs, -band, band, color="#2874a6", alpha=0.18,
                label="MC 1$\\sigma$ (400,000 particles)")
ax.set_xscale("log")
ax.set_xlabel("Aspect ratio  AR = H / D", fontsize=12)
ax.set_ylabel("MC minus reference (%)", fontsize=12)
ax.set_ylim(-3, 3)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)
ax.set_title("(b) Residual of the same comparison\n"
             "agreement is within Monte Carlo statistics", fontsize=12.5)

# ---- (c) Pi collapse in 3D -------------------------------------------
ax = axes[2]
famA = [(10, 0.040), (20, 0.020), (40, 0.010)]        # AR * s0 = 0.4
famB = [(10, 0.040), (20, 0.010), (40, 0.0025)]       # AR * sqrt(s0) = 2.0
for (AR, s0), c in zip(famA, ["#e59866", "#dc7633", "#a04000"]):
    ax.plot(shape(float(AR), s0), zc[:-1], lw=1.7, ls="--", color=c,
            label=f"$AR\\,s_0$ = 0.4:  AR = {AR}")
for (AR, s0), c in zip(famB, ["#7fb3d5", "#2874a6", "#154360"]):
    ax.plot(shape(float(AR), s0), zc[:-1], lw=2.4, color=c,
            label=f"$AR\\sqrt{{s_0}}$ = 2.0:  AR = {AR}")
ax.invert_yaxis()
ax.set_xlim(0, 1.05)
ax.set_xlabel(r"Normalized coverage  $\theta(z)\,/\,\theta(0)$", fontsize=12)
ax.set_ylabel("Normalized depth  z / H", fontsize=12)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower right")
ax.set_title(r"(c) Shape collapse in the cylinder"
             "\n"
             r"$\Pi_1 = AR\sqrt{s_0}$ collapses, $AR\,s_0$ does not", fontsize=12.5)

fig.suptitle("Geometry decision: 3D axisymmetric cylinder — validation",
             fontsize=15, y=1.02)
fig.savefig("geometry_validation.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("saved geometry_validation.png")
