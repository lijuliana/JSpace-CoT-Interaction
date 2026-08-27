"""Figure F1: writing is not a capacity-triggered fallback.

(a) Silent accuracy collapses after one to two dependent steps while
writing-allowed accuracy stays high. (b) The fraction of intermediate
values written is at ceiling from depth 1. (c) Change in accuracy under
each intervention.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from style import (apply, tag, wilson, load, save,
                   BLUE, VERM, GRAY, MUT)

apply()


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

wrt = []
for d in d671:
    k = n = 0
    for r in r671:
        if r["condition"] != "free" or r["difficulty"] != d:
            continue
        for rec in (r.get("externalization") or {}).get("records", []):
            n += 1
            k += bool(rec["written"])
    wrt.append((k, n))

fig = plt.figure(figsize=(5.5, 3.35))
gs1 = fig.add_gridspec(1, 2, wspace=0.28, left=0.09, right=0.99,
                       top=0.93, bottom=0.615)
gs2 = fig.add_gridspec(1, 1, left=0.26, right=0.99, top=0.445, bottom=0.115)
axA, axB = [fig.add_subplot(gs1[i]) for i in range(2)]
axC = fig.add_subplot(gs2[0])

# (a) silent collapse
xA = np.log2(d671)
axA.plot(xA, free_acc, "-o", color=BLUE, ms=3.2, lw=1.2, zorder=3)
axA.plot(xA, sil_acc, "-o", color=VERM, ms=3.2, lw=1.2, zorder=3)
axA.plot(np.log2(d7b), sil7b, "--^", color=VERM, ms=3.2, lw=1.0,
         alpha=0.65, zorder=2)
axA.text(2.2, 1.06, "Writing allowed (R1-671B)", color=BLUE, fontsize=6.4)
axA.text(1.55, 0.35, "Silent,\nR1-671B", color=VERM, fontsize=6.4)
axA.text(0.62, 0.66, "Silent,\nR1-distill-7B", color=VERM, fontsize=6.4,
         alpha=0.75)
axA.set_xlabel("Dependent steps $d$")
axA.set_ylabel("Accuracy")
axA.set_xticks(xA)
axA.set_xticklabels(d671)
axA.set_xlim(-0.4, 6.4)
axA.set_ylim(-0.04, 1.14)
axA.set_yticks([0, 0.5, 1.0])
tag(axA, "a", x=-0.16)

# (b) writing at ceiling
m = [k / n for k, n in wrt]
lo = [max(0.0, m[i] - wilson(k, n)[0]) for i, (k, n) in enumerate(wrt)]
hi = [max(0.0, wilson(k, n)[1] - m[i]) for i, (k, n) in enumerate(wrt)]
axB.errorbar(xA, m, yerr=[lo, hi], fmt="o", color=BLUE, ms=3.4,
             elinewidth=1.0, capsize=2, zorder=3)
axB.set_xlabel("Dependent steps $d$")
axB.set_ylabel("Intermediate values\nwritten out (fraction)")
axB.set_xticks(xA)
axB.set_xticklabels(d671)
axB.set_xlim(-0.4, 6.4)
axB.set_ylim(-0.04, 1.14)
axB.set_yticks([0, 0.5, 1.0])
axB.text(3.2, 0.50, "30 to 1,920 values per\ndepth; no onset threshold",
         fontsize=6.2, color=MUT, ha="center")
tag(axB, "b", x=-0.16)

# (c) interventions
rows = [
    ("J-space ablation", -0.04, VERM),
    ("Random directions", -0.04, GRAY),
    ("Token cap 256", -0.07, BLUE),
    ("Residual lesion, chains", -0.11, VERM),
    ("Residual lesion, box tracking", -0.34, VERM),
    ("Token cap 128", -1.00, BLUE),
]
yC = np.arange(len(rows))[::-1]
axC.axvline(0, color="#cccccc", lw=0.7, zorder=0)
for (lab, v, c), yy in zip(rows, yC):
    axC.plot([0, v], [yy, yy], color=c, lw=0.9, alpha=0.4, zorder=2)
    axC.plot(v, yy, "o", color=c, ms=4.0, zorder=3)
axC.set_yticks(yC)
axC.set_yticklabels([r[0] for r in rows], fontsize=6.6)
axC.set_xlabel("Change in accuracy")
axC.set_xlim(-1.1, 0.12)
axC.set_xticks([-1.0, -0.5, 0.0])
axC.set_ylim(-0.6, 5.6)
axC.text(-0.05, 1.34, "Positive control: internally held task",
         fontsize=6.0, color=MUT, ha="right")
axC.text(-0.97, 0.36, "Truncation, not internalization",
         fontsize=6.0, color=MUT, ha="left")
tag(axC, "c", x=-0.31)

for ax in (axA, axB):
    ax.grid(axis="y", color="#efefef", lw=0.6)
    ax.set_axisbelow(True)
axC.grid(axis="x", color="#efefef", lw=0.6)
axC.set_axisbelow(True)

save(fig, "F1_capacity")
