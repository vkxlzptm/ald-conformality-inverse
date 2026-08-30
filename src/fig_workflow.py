"""Workflow diagram: how a measurement becomes a set of surface-reaction
parameters, and what the conventional route costs instead.

Replaces the earlier mermaid source, which needed a renderer to view.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "workflow.png")

INK = "#17202a"
BLUE, GREEN, ORANGE, RED, PURPLE, GREY = ("#2874a6", "#1e8449", "#b9770e",
                                          "#c0392b", "#7d3c98", "#566573")
FILL = {BLUE: "#eaf2f8", GREEN: "#e8f6ef", ORANGE: "#fdf2e2",
        RED: "#fdedec", PURPLE: "#f4ecf7", GREY: "#eceff1"}


def box(ax, x, y, w, h, edge, title, body="", ts=11, bs=9.0, fill=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                fc=fill or FILL[edge], ec=edge, lw=1.8,
                                zorder=3))
    ax.text(x + w / 2, y + h - 0.024, title, ha="center", va="top",
            fontsize=ts, color=INK, fontweight="bold", zorder=4)
    if body:
        ax.text(x + w / 2, y + h - 0.058, body, ha="center", va="top",
                fontsize=bs, color=GREY, zorder=4, linespacing=1.42)


def arr(ax, x0, y0, x1, y1, col=INK, lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=17, lw=lw, color=col,
                                 linestyle=ls, zorder=2, shrinkA=0, shrinkB=0))


def band(ax, y, h, col, num, name, cost):
    ax.add_patch(Rectangle((0.055, y), 0.935, h, fc="none", ec=col, lw=1.0,
                           ls=(0, (4, 4)), alpha=0.55, zorder=1))
    ax.text(0.048, y + h / 2, f"{num}", ha="right", va="center", fontsize=21,
            color=col, fontweight="bold")
    ax.text(0.066, y + h + 0.010, name, ha="left", va="bottom", fontsize=12.5,
            color=col, fontweight="bold")
    ax.text(0.982, y + h + 0.010, cost, ha="right", va="bottom", fontsize=10.5,
            color=col, style="italic")


fig, ax = plt.subplots(figsize=(17.6, 14.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

BH, IH, PAD = 0.183, 0.160, 0.011

# ------------------------------------------------------------------ band 1
Y = 0.762
band(ax, Y, BH, BLUE, "1", "Training-data generation",
     "offline, once  ·  6 h on 6 cores")
yb = Y + PAD
box(ax, 0.075, yb, 0.185, IH, BLUE, "Sample the prior",
    "$s_0$ at $T_{ref}$ · $E_a$\nsteric exponent n\nre-emission survival\nsite density")
box(ax, 0.295, yb, 0.235, IH, BLUE, "Ballistic MC simulator",
    "3D axisymmetric cylinder\nfree-molecular transport\ndiffuse re-emission\n"
    "self-limiting surface reaction")
box(ax, 0.565, yb, 0.185, IH, BLUE, "Profiles",
    "3 temperatures\n× 64 depth bins\none run, snapshotted\nat 6 dose levels")
box(ax, 0.785, yb, 0.185, IH, BLUE, "324,000 wafers",
    "108,000 draws,\neach split into its\nthree temperatures\n600 shards,\nre-running = resuming")
for x in (0.262, 0.532, 0.752):
    arr(ax, x, yb + IH / 2, x + 0.030, yb + IH / 2, BLUE)
ax.text(0.5, Y - 0.014,
        "written from scratch — this is not kinetic Monte Carlo, and is not called that",
        ha="center", va="top", fontsize=10, color=RED, style="italic")

# ------------------------------------------------------------------ band 2
Y = 0.522
band(ax, Y, BH, GREEN, "2", "Training", "offline, once  ·  35 min")
yb = Y + PAD
box(ax, 0.075, yb, 0.215, IH, ORANGE, "Split by shard",
    "never by example —\nthe 6 dose snapshots of\none draw share the same\n"
    "ground truth, so an\nexample-level split leaks")
box(ax, 0.325, yb, 0.215, IH, GREEN, "Measurement model",
    "applied fresh every epoch\nnoise topped up to ±3 %\n10 of 64 bins masked\n"
    "(doubles as augmentation)")
box(ax, 0.575, yb, 0.395, IH, GREEN, "1D CNN  →  mixture-density head",
    "in:   3 × 64 profile channels  and  log AR.  Nothing else.\n"
    "out:  $(s_0,\ n,\ $re-emission$,\ \Pi_2)$, as a posterior\n"
    "every input and output is dimensionless\n"
    "the dose is not an input: the profile sees it only through $\Pi_2$")
for x in (0.292, 0.542):
    arr(ax, x, yb + IH / 2, x + 0.030, yb + IH / 2, GREEN)

# ------------------------------------------------------------------ band 3
Y = 0.282
band(ax, Y, BH, PURPLE, "3", "Inference", "online  ·  0.13 ms per wafer")
yb = Y + PAD
box(ax, 0.075, yb, 0.205, IH, PURPLE, "One wafer",
    "one TEM cross-section\nat one temperature\nthickness normalised by\nthe flat-top thickness")
box(ax, 0.315, yb, 0.185, IH, PURPLE, "Single forward pass")
box(ax, 0.535, yb, 0.215, IH, PURPLE, "Units restored outside",
    "$n_s = N_{dose}/(4\,AR\,\Pi_2)$\n$E_a$ from ln $s_0$ vs $1/T$\nacross a temperature split\nclosed form — no simulator")
box(ax, 0.785, yb, 0.185, IH, PURPLE, "Process prescription",
    "dose required at a\ndifferent aspect ratio\nobserve AR 20\n→ prescribe for AR 50")
for x in (0.282, 0.502, 0.752):
    arr(ax, x, yb + IH / 2, x + 0.030, yb + IH / 2, PURPLE)

# --------------------------------------------------------------- band 4
Y = 0.030
box(ax, 0.075, Y, 0.400, 0.190, RED, "What this replaces",
    "Derivative-free least squares, re-run for every measurement\n"
    "Nelder-Mead · multi-start · common random numbers\n"
    "~870 simulator calls   ·   ~12 min per measurement\n"
    "point estimate only, with no statement of uncertainty\n"
    "the residual reaches the noise floor while the parameters\n"
    "stay far from truth — the problem is partly degenerate",
    ts=12, bs=9.6)
box(ax, 0.520, Y, 0.450, 0.190, GREY, "How it is checked",
    "1.  Zero-reaction transmission vs a deterministic solution of the\n"
    "     Clausing integral equation — agrees to ≤ 0.25 %   ·  PASSED\n"
    "2.  Head to head with least squares, same benchmark cases\n"
    "3.  Leakage audit — can the conditions alone give a parameter away?\n"
    "4.  Aspect-ratio transfer — observe AR 20, predict the dose at AR 50\n"
    "5.  Model misspecification — mechanisms outside the training prior",
    ts=12, bs=9.6, fill="#eef2f7")

arr(ax, 0.5, 0.762, 0.5, 0.708, BLUE)
arr(ax, 0.5, 0.522, 0.5, 0.468, GREEN)
arr(ax, 0.275, 0.282, 0.275, 0.226, PURPLE)
arr(ax, 0.745, 0.282, 0.745, 0.226, PURPLE)
ax.text(0.288, 0.252, "replaces", fontsize=10.5, color=RED, style="italic")
ax.text(0.758, 0.252, "is checked by", fontsize=10.5, color=GREY, style="italic")

fig.suptitle("ALD step coverage — inverse problem by amortized inference",
             fontsize=19, y=0.982)
fig.text(0.5, 0.962,
         "the cost of fitting moves from measurement time to training time; "
         "the physics is not replaced",
         ha="center", va="top", fontsize=12, color=GREY, style="italic")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print("saved ->", OUT)
