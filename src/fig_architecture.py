"""Diagram of the amortized inference network.

Left: what goes in and how the 1D convolutional trunk reduces it.
Right: what the mixture-density head emits, and why a distribution rather than
a single number.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "network_architecture.png")

INK = "#1b2631"
BLUE, GREEN, ORANGE, GREY = "#2874a6", "#1e8449", "#b9770e", "#5d6d7e"


def box(ax, x, y, w, h, face, edge, title, sub="", tsize=10.5, ssize=8.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=face, ec=edge, lw=1.7, zorder=3))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center",
            va="center", fontsize=tsize, color=INK, fontweight="bold",
            zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=ssize, color=GREY, zorder=4)


def arrow(ax, x0, y0, x1, y1, col=INK, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=15, lw=lw, color=col,
                                 zorder=2, shrinkA=0, shrinkB=0))


fig = plt.figure(figsize=(18.0, 9.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.66, 1.0], wspace=0.09)
axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1])
for a in (axL, axR):
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")

# ----------------------------------------------------------------- input
zc = np.linspace(0, 1, 64)
prof = [1 - 0.80 * zc ** 1.6]
cols = ["#2874a6"]
labs = ["one wafer"]
ax_in = axL.inset_axes([0.025, 0.775, 0.175, 0.195])
for p_, c, l in zip(prof, cols, labs):
    ax_in.plot(p_, zc, lw=2.2, color=c, label=l)
rng = np.random.default_rng(3)
miss = rng.choice(64, 10, replace=False)
ax_in.plot(prof[0][miss], zc[miss], "x", ms=5, color="#c0392b")
ax_in.invert_yaxis(); ax_in.set_xlim(0, 1.05)
ax_in.set_xlabel("Coverage  " + r"$\theta$", fontsize=8.5)
ax_in.set_ylabel("Depth  z / H", fontsize=8.5)
ax_in.tick_params(labelsize=7.5)
ax_in.legend(fontsize=6.6, loc="upper left", frameon=False)
ax_in.set_title("One measurement", fontsize=9.5)
axL.text(0.112, 0.700, "one temperature  ×  64 depth bins\n"
         r"$\times$ marks the 10 bins a measurement misses",
         ha="center", va="top", fontsize=8.6, color=GREY)

box(axL, 0.255, 0.775, 0.128, 0.195, "#eaf2f8", BLUE, "Input tensor",
    "3 × 64", 11, 9.5)
arrow(axL, 0.212, 0.872, 0.252, 0.872, BLUE)

trunk = [("Conv 5, s1", "3 → 96\nlength 64"),
         ("Conv 5, s2", "96 → 96\nlength 32"),
         ("Conv 5, s2", "96 → 192\nlength 16"),
         ("Conv 5, s2", "192 → 192\nlength 8")]
x = 0.435
for t, sub in trunk:
    box(axL, x, 0.775, 0.125, 0.195, "#e8f6ef", GREEN, t, sub, 10, 8.6)
    arrow(axL, x - 0.023, 0.872, x - 0.003, 0.872, GREEN)
    x += 0.138
axL.text(0.71, 0.995, "Convolutional trunk  —  BatchNorm + GELU after each",
         ha="center", fontsize=10.5, color=GREEN, fontweight="bold")

axL.text(0.60, 0.740,
         "3 input channels  =  coverage  +  log coverage  +  mask.  The log channel\n"
         "keeps small values deep in the feature alive; the mask channel separates\n"
         "'not measured' from 'bare surface'.  Convolution runs over depth.",
         ha="center", va="top", fontsize=8.8, color=GREY)

# --------------------------------------------------------------- bottom row
YB, HB = 0.40, 0.155
box(axL, 0.035, YB, 0.185, HB, "#fdf2e2", ORANGE, "Known condition",
    "log AR   (that is all)", 10.5, 9)
box(axL, 0.285, YB, 0.175, HB, "#fdf2e2", ORANGE, "Global pooling",
    "mean ⊕ max  →  384", 10.5, 9)
box(axL, 0.525, YB, 0.175, HB, "#f4ecf7", "#7d3c98", "Dense layers",
    "385 → 384 → 384,  GELU", 10.5, 9)
box(axL, 0.765, YB, 0.205, HB, "#fdedec", "#c0392b", "Mixture-density head",
    "8 components × 4 parameters", 10.5, 8.8)

arrow(axL, 0.935, 0.770, 0.935, 0.605, GREEN)
arrow(axL, 0.935, 0.605, 0.375, 0.605, GREEN)
arrow(axL, 0.375, 0.605, 0.375, 0.560, GREEN)
arrow(axL, 0.222, YB + HB / 2, 0.282, YB + HB / 2, ORANGE)
arrow(axL, 0.462, YB + HB / 2, 0.522, YB + HB / 2, ORANGE)
arrow(axL, 0.702, YB + HB / 2, 0.762, YB + HB / 2, "#7d3c98")

axL.text(0.1275, 0.375, "the dose is not an input: the profile\n"
         "depends on it only through $\\Pi_2$, which is unknown",
         ha="center", va="top", fontsize=8.4, color="#a04000")

axL.text(0.5, 0.245,
         "649,512 parameters   ·   one forward pass is milliseconds",
         ha="center", fontsize=11, color=INK)
axL.text(0.5, 0.175,
         "Everything in and out is dimensionless.  Units are restored afterwards in "
         "closed form:\n"
         "$n_s = N_{dose}\\,/\\,(4\\,AR\\,\\Pi_2)$   and   "
         "$E_a$ from the slope of $\\ln s_0$ against $1/T$ across a temperature split.",
         ha="center", va="top", fontsize=9.6, color=INK)
axL.set_title("(a) From one measurement to a posterior", fontsize=13.5,
              loc="left", pad=6)

# ------------------------------------------------------- right: the head
box(axR, 0.06, 0.845, 0.24, 0.10, "#fdedec", "#c0392b",
    "mixing weights  π", "8 numbers, softmax", 10, 8.4)
box(axR, 0.37, 0.845, 0.26, 0.10, "#fdedec", "#c0392b",
    "means  μ", "8 × 4", 10, 8.4)
box(axR, 0.70, 0.845, 0.24, 0.10, "#fdedec", "#c0392b",
    "widths  σ", "8 × 4", 10, 8.4)

ax_p = axR.inset_axes([0.12, 0.26, 0.76, 0.50])
gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 220), np.linspace(-2.6, 2.6, 220))
ridge = np.exp(-((gy - 0.62 * gx) ** 2) / 0.16 - (gx ** 2) / 3.1)
ax_p.contourf(gx, gy, ridge, levels=18, cmap="Blues")
ax_p.plot(0.35, 0.28, "*", ms=19, color="#f4d03f", mec="k", mew=0.8)
ax_p.annotate("truth", (0.35, 0.28), xytext=(10, -16),
              textcoords="offset points", fontsize=10, fontweight="bold")
ax_p.plot(-1.35, -1.5, "o", ms=9, color="#c0392b")
ax_p.annotate("least-squares\npoint estimate", (-1.35, -1.5), xytext=(6, -30),
              textcoords="offset points", fontsize=9, color="#c0392b")
ax_p.set_xlabel(r"log $s_0$   (normalized)", fontsize=9.5)
ax_p.set_ylabel(r"log $\Pi_2$   (normalized)", fontsize=9.5)
ax_p.tick_params(labelsize=8)
ax_p.set_title("Posterior for one measurement, schematic", fontsize=10)

axR.text(0.5, 0.185,
         "The degeneracy is a ridge, not a blob: raising the sticking\n"
         "probability and the dose per site together leaves the profile\n"
         "almost unchanged. A single number cannot say that; the width\n"
         "and tilt of the posterior can.",
         ha="center", va="top", fontsize=9.4, color=INK)
axR.text(0.5, 0.035,
         "Diagonal components can only draw axis-aligned ellipses, so a tilted\n"
         "ridge costs several of them — a full-covariance head is the upgrade.",
         ha="center", va="top", fontsize=8.4, color=GREY, style="italic")
axR.set_title("(b) What the head emits, and why", fontsize=13.5, loc="left",
              pad=2)

fig.suptitle("Amortized inference network for ALD step coverage",
             fontsize=16, y=0.985)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print("saved ->", OUT)
