"""Figure F1: writing is not a capacity-triggered fallback.

Panel A: silent (no chain of thought) accuracy collapses after one to two
dependent steps while writing-allowed accuracy stays high.
Panel B: fraction of intermediate values written per depth, at ceiling from
d=1 (dots with Wilson intervals; the raw string matcher gives 1.00 at every
depth, so the display carries n via interval width).
Panel C: change in accuracy under each intervention; nothing moves state
between the workspace and the trace except cutting the text channel below
the floor.
"""

import json
import math
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analysis.figstyle import apply

apply()
BLUE, VERM, GREEN, SKY = "#0072B2", "#D55E00", "#009E73", "#56B4E9"
INK, MUT, GRAY = "#1a1a1a", "#666666", "#8a8a8a"
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def load(name):
    with open(os.path.join(ROOT, "results", "raw", name)) as f:
        return [json.loads(l) for l in f]


def reextract(rec):
    """Fix parser misses where pred is a non-numeric token: take the model's
    stated final answer (**Answer:** N or the last integer in the trace)."""
    if re.fullmatch(r"-?\d+", str(rec["pred"])):
        return rec["correct"]
    m = re.findall(r"\*\*Answer:?\*\*:?\s*(-?\d+)", rec["trace"])
    if not m:
        m = re.findall(r"(-?\d+)", rec["trace"])
    return bool(m) and m[-1] == str(rec["answer"])


r671 = load("p1_var_r1_671b.jsonl")
d671 = [1, 2, 4, 8, 16, 32, 64]
free_acc = [np.mean([reextract(r) for r in r671
                     if r["condition"] == "free" and r["difficulty"] == d])
            for d in d671]
sil_acc = [np.mean([r["correct"] for r in r671
                    if r["condition"] == "direct" and r["difficulty"] == d])
           for d in d671]

r7b = load("p1_var_7b.jsonl")
d7b = [1, 2, 4, 8, 16, 32, 48]
sil7b = [np.mean([r["correct"] for r in r7b
                  if r["condition"] == "direct" and r["difficulty"] == d])
         for d in d7b]

# Panel B: writing rate per depth (R1-671B, free condition)
wrt = []
for d in d671:
    k = n = 0
    for r in r671:
        if r["condition"] != "free" or r["difficulty"] != d:
            continue
        ext = r.get("externalization") or {}
        for rec in ext.get("records", []):
            n += 1
            k += bool(rec["written"])
    wrt.append((k, n))

fig = plt.figure(figsize=(5.6, 2.05))
gs1 = fig.add_gridspec(1, 2, wspace=0.30, left=0.075, right=0.56,
                       top=0.86, bottom=0.21)
gs2 = fig.add_gridspec(1, 1, left=0.775, right=0.99, top=0.86, bottom=0.21)
axA, axB = [fig.add_subplot(gs1[i]) for i in range(2)]
axC = fig.add_subplot(gs2[0])

# ---- Panel A ----
xA = np.log2(d671)
axA.axvspan(-0.5, 1.0, color="#f0f0f0", zorder=0)
axA.plot(xA, free_acc, "-o", color=BLUE, ms=3.4, lw=1.4, zorder=3)
axA.plot(xA, sil_acc, "-o", color=VERM, ms=3.4, lw=1.4, zorder=3)
axA.plot(np.log2(d7b), sil7b, "--o", color=SKY, ms=3.0, lw=1.2, zorder=2)
axA.text(2.1, 1.045, "writing allowed (R1-671B)", color=BLUE, fontsize=6.0)
axA.text(1.35, 0.33, "silent,\nR1-671B", color=VERM, fontsize=6.0)
axA.text(0.75, 0.62, "silent,\nR1-distill-7B", color=SKY, fontsize=6.0)
axA.set_title("silent accuracy collapses\nafter one to two steps", fontsize=6.8)
axA.set_xlabel("dependent steps")
axA.set_ylabel("accuracy")
axA.set_xticks(np.log2(d671))
axA.set_xticklabels(d671)
axA.set_xlim(-0.4, 6.4)
axA.set_ylim(-0.04, 1.12)
axA.set_yticks([0, 0.5, 1.0])

# ---- Panel B ----
m = [k / n for k, n in wrt]
lo = [max(0.0, m[i] - wilson(k, n)[0]) for i, (k, n) in enumerate(wrt)]
hi = [max(0.0, wilson(k, n)[1] - m[i]) for i, (k, n) in enumerate(wrt)]
axB.axvspan(-0.5, 1.0, color="#f0f0f0", zorder=0)
axB.errorbar(xA, m, yerr=[lo, hi], fmt="o", color=BLUE, ms=3.6,
             elinewidth=1.1, capsize=2, zorder=3)
axB.set_title("every intermediate value is\nwritten, from step one on", fontsize=6.8)
axB.set_xlabel("dependent steps")
axB.set_ylabel("fraction of values written")
axB.set_xticks(np.log2(d671))
axB.set_xticklabels(d671)
axB.set_xlim(-0.4, 6.4)
axB.set_ylim(-0.04, 1.12)
axB.set_yticks([0, 0.5, 1.0])
axB.text(3.3, 0.52, "30 to 1,920 values\nper depth; no onset\nthreshold anywhere",
         fontsize=5.8, color=MUT, ha="center")

# ---- Panel C ----
rows = [
    ("J-space ablation", -0.04, GRAY),
    ("random directions", -0.04, GRAY),
    ("token cap 256", -0.07, BLUE),
    ("residual lesion, chains", -0.11, VERM),
    ("residual lesion, box tracking", -0.34, VERM),
    ("token cap 128", -1.00, BLUE),
]
yC = np.arange(len(rows))[::-1]
axC.axvline(0, color="#cccccc", lw=0.8, zorder=0)
for (lab, v, c), yy in zip(rows, yC):
    axC.plot([0, v], [yy, yy], color=c, lw=1.0, alpha=0.45, zorder=2)
    axC.plot(v, yy, "o", color=c, ms=4.2, zorder=3)
axC.set_yticks(yC)
axC.set_yticklabels([r[0] for r in rows], fontsize=6.0)
axC.set_xlabel("change in accuracy")
axC.set_title("no intervention moves state\nbetween workspace and trace", fontsize=6.8)
axC.set_xlim(-1.1, 0.12)
axC.set_xticks([-1.0, -0.5, 0.0])
axC.text(-0.05, 1.30, "positive control: internally held task",
         fontsize=5.4, color=MUT, ha="right")
axC.text(-0.95, 0.32, "truncation, not internalization",
         fontsize=5.4, color=MUT, ha="left")
for ax in (axA, axB, axC):
    ax.grid(axis="y" if ax is not axC else "x", color="#ececec", lw=0.7)
    ax.set_axisbelow(True)

out = os.path.join(os.path.dirname(__file__), "..")
fig.savefig(os.path.join(out, "F1_capacity.pdf"))
fig.savefig("/tmp/cur_F1_capacity.png", dpi=170)
print("F1 done", free_acc[0], m)
