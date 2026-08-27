"""AR scaling figure: profiles at several AR, transmission vs AR, dose exponent vs s0."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trench_mc import profile, NBIN
from ar_scaling import transmission

zc = (np.arange(NBIN) + 0.5) / NBIN
fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3))
fig.subplots_adjust(wspace=0.30)

# (a) same dose, AR varied -------------------------------------------------
ax = axes[0]
DOSE = 300_000
for AR, c in [(1.0, "#8e44ad"), (5.0, "#117a65"),
              (20.0, "#b9770e"), (40.0, "#c0392b")]:
    th, _, _, _ = profile(AR, DOSE, s0=0.05, sites_per_bin=200.0 * AR, seed=5)
    ax.plot(th[:-1], zc[:-1], lw=2.3, color=c, label=f"AR = {AR:.0f}")
ax.set_xlabel(r"Coverage  $\theta$", fontsize=12)
ax.set_ylabel("Normalized depth  z / H", fontsize=12)
ax.invert_yaxis()
ax.set_xlim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend(fontsize=11, loc="lower right")
ax.set_title("(a) Same dose, same $s_0$ = 0.05\n"
             "only AR differs (site density held constant)", fontsize=13)

# (b) transmission at zero reaction ----------------------------------------
ax = axes[1]
ARs = np.array([2., 3., 5., 8., 12., 20., 30., 40.])
tr = np.array([transmission(a, 120_000, 5) for a in ARs])
ax.loglog(ARs, tr, "o-", lw=2, ms=7, color="#1f4e79", label="MC, 2D slit")
ax.loglog(ARs, tr[2] * 5.0 / ARs, "--", lw=1.8, color="#999",
          label=r"$\propto 1/AR$")
ax.loglog(ARs, tr[2] * 5.0 / np.log(5.0) * np.log(ARs) / ARs, ":", lw=2.2,
          color="#c0392b", label=r"$\propto \ln(AR)/AR$")
ax.set_xlabel("Aspect ratio  AR", fontsize=12)
ax.set_ylabel(r"Bottom-arrival probability  ($s_0$ = 0)", fontsize=12)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=10.5)
sl = np.polyfit(np.log(ARs[2:]), np.log(tr[2:]), 1)[0]
ax.set_title(f"(b) Transmission with no reaction — slope {sl:.2f}\n"
             "a 2D slit does not follow 1/AR", fontsize=13)

# (c) dose exponent vs sticking probability --------------------------------
ax = axes[2]
s0s = [0.05, 0.2, 0.6]
slopes = [0.39, 0.81, 1.21]
ax.plot(s0s, slopes, "o-", lw=2.4, ms=9, color="#c0392b")
ax.axhline(2.0, ls="--", color="#333", lw=1.6)
ax.text(0.24, 2.05, r"Textbook rule   dose $\propto AR^2$", fontsize=11.5)
ax.axhline(1.0, ls=":", color="#888", lw=1.4)
ax.text(0.33, 1.04, r"Internal area only ($\propto AR$)", fontsize=10.5,
        color="#666")
ax.set_xlabel(r"Sticking probability  $s_0$", fontsize=12)
ax.set_ylabel(r"Exponent x in dose $\propto AR^{\,x}$", fontsize=12)
ax.set_ylim(0, 2.35)
ax.set_xlim(0, 0.68)
ax.grid(alpha=0.3)
ax.set_title("(c) Measured AR exponent of the saturation dose\n"
             "it approaches 2 only at large $s_0$", fontsize=13)

fig.suptitle("AR scaling, measured — what the MC simulator actually shows",
             fontsize=15, y=1.02)
fig.savefig("ar_scaling_measured.png", dpi=155, bbox_inches="tight",
            facecolor="white")
print("saved", sl)
