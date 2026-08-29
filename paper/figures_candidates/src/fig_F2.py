"""Figure F2: written positions are settable, compositional registers.

(A) Schematic of the state patch (same visible text, different stored
value). (B) Per-model rates with 95% intervals. (C, D) Answer category by
planted condition for the two-register composition test.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from style import (apply, tag, load, save,
                   BLUE, VERM, GREEN, GRAY, MUT, INK, LIGHT,
                   T_GRAY, T_GREEN, T_BLUE)

apply()


def cats(fname):
    """Stacked-bar decomposition uses collision-free items; the direct label
    uses the hit rate over all items so it matches the numbers in the text."""
    conds = ["none", "aonly", "bonly", "joint"]
    recs = load(fname)
    out = {c: {"clean": 0, "apB": 0, "Abp": 0, "apbp": 0, "other": 0, "n": 0}
           for c in conds}
    for r in recs:
        exp = {"clean": r["A"] + r["B"], "apB": r["ap"] + r["B"],
               "Abp": r["A"] + r["bp"], "apbp": r["ap"] + r["bp"]}
        if len(set(exp.values())) < 4:
            continue
        rev = {str(v): k for k, v in exp.items()}
        for c in conds:
            for a in r[c]["answers"]:
                out[c][rev.get(a, "other")] += 1
                out[c]["n"] += 1
    hits = {c: sum(r[c]["hit"] for r in recs) / len(recs) for c in conds}
    for c in conds:
        n = out[c].pop("n")
        for k in out[c]:
            out[c][k] /= n
    return out, hits


def band(ax, x, y, w, h, text, fc, tc=INK, fs=6.4):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, family="monospace")


fig = plt.figure(figsize=(5.5, 4.15))
gsA = fig.add_gridspec(1, 1, left=0.035, right=0.425, top=0.955, bottom=0.60)
gsD = fig.add_gridspec(1, 1, left=0.615, right=0.995, top=0.955, bottom=0.60)
gsB = fig.add_gridspec(1, 2, wspace=0.13, left=0.10, right=0.995,
                       top=0.425, bottom=0.16)
axS = fig.add_subplot(gsA[0])
axD = fig.add_subplot(gsD[0])
axQ1, axQ2 = fig.add_subplot(gsB[0]), fig.add_subplot(gsB[1])

# (A) schematic: even vertical rhythm, self-contained wording
axS.axis("off")
axS.set_xlim(0, 1)
axS.set_ylim(0, 1)
axS.text(0.5, 0.955, "The visible reasoning prefix; every\ncharacter stays exactly in place:",
         ha="center", va="center", fontsize=6.6, color=MUT)
band(axS, 0.10, 0.76, 0.80, 0.12, "After step 5 the value is 457.", T_GRAY,
     fs=6.6)
axS.text(0.5, 0.665, "Activation at the 457 position:",
         ha="center", va="center", fontsize=6.6, color=MUT)
band(axS, 0.16, 0.46, 0.20, 0.12, "457", T_GRAY, fs=6.6)
band(axS, 0.64, 0.46, 0.20, 0.12, "462", T_GREEN, tc=GREEN, fs=6.6)
axS.add_patch(FancyArrowPatch((0.39, 0.52), (0.61, 0.52), arrowstyle="->",
                              color=GREEN, lw=1.2, mutation_scale=9))
axS.text(0.5, 0.385, "patched from a run that wrote 462;\n462 appears in no text",
         ha="center", va="center", fontsize=6.0, color=GREEN)
axS.text(0.5, 0.27, "The model keeps generating and its\nfinal answer builds on 462, not 457:",
         ha="center", va="center", fontsize=6.6, color=MUT)
band(axS, 0.06, 0.07, 0.88, 0.12, "final answer = chain continued from 462",
     T_BLUE, tc=BLUE, fs=6.4)
tag(axS, "a", x=0.0, y=0.97)

# (B) per-model rates
models = ["Qwen3-4B", "Qwen2.5-7B", "Phi-3-medium", "OLMo-2-7B",
          "R1-distill-7B", "R1-distill-14B"]
restore = [(1.00, 1.00, 1.00), (0.97, 0.92, 1.00), (0.98, 0.95, 1.00),
           (0.88, 0.82, 0.94), (1.00, 0.99, 1.00), (0.99, 0.98, 1.00)]
plant = [(1.00, 0.99, 1.00), (0.74, 0.67, 0.81), (0.94, 0.91, 0.97),
         (0.93, 0.88, 0.97), (0.76, 0.68, 0.84), (0.70, 0.61, 0.80)]
rand = [(0.00, 0.00, 0.00), (0.00, 0.00, 0.01), (0.00, 0.00, 0.00),
        (0.00, 0.00, 0.00), (0.30, 0.22, 0.39), (0.21, 0.13, 0.29)]
y = np.arange(6)[::-1]
for data, color, mk, label, o in [
        (plant, BLUE, "o", "Inject a new value", 0.0),
        (restore, GREEN, "s", "Restore the correct value", 0.24),
        (rand, GRAY, "^", "Random direction (control)", -0.24)]:
    m = [d[0] for d in data]
    lo = [d[0] - d[1] for d in data]
    hi = [d[2] - d[0] for d in data]
    axD.errorbar(m, y + o, xerr=[lo, hi], fmt=mk, color=color, ms=4.2,
                 mec="white", mew=0.8, elinewidth=1.0, label=label, zorder=3)
axD.set_yticks(y)
axD.set_yticklabels(models, fontsize=7.0)
axD.set_xlabel("Fraction of runs where the final\nanswer uses the injected value",
               fontsize=7)
axD.set_xlim(-0.04, 1.06)
axD.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axD.set_xticklabels(["0", "", "0.5", "", "1"])
axD.grid(axis="x", color="#efefef", lw=0.6)
axD.set_axisbelow(True)
axD.legend(fontsize=6.6, loc="upper left", bbox_to_anchor=(0.08, 1.04))
tag(axD, "b", x=-0.36)

# (C, D) composition answer categories
CAT = [("clean", GRAY, "Correct sum $A{+}B$"), ("apB", VERM, "$a'{+}B$"),
       ("Abp", GREEN, "$A{+}b'$"), ("apbp", BLUE, "$a'{+}b'$"),
       ("other", LIGHT, "Other")]
DOM = {"none": "clean", "aonly": "apB", "bonly": "Abp", "joint": "apbp"}
conds = ["none", "aonly", "bonly", "joint"]
xt = ["No\ninjection", "Inject\n$a'$", "Inject\n$b'$", "Inject\nboth"]
for ax, fname, mname, letter in [
        (axQ1, "s1_qwen25-7b.jsonl", "Qwen2.5-7B", "c"),
        (axQ2, "s1_qwen3-4b.jsonl", "Qwen3-4B", "d")]:
    data, hits = cats(fname)
    for xi, c in enumerate(conds):
        bot = 0.0
        for key, color, _ in CAT:
            v = data[c][key]
            if v > 0:
                ax.bar(xi, v, 0.58, bottom=bot, color=color, linewidth=0)
            bot += v
        dk = DOM[c]
        dcol = dict((k, col) for k, col, _ in CAT)[dk]
        dlab = dict((k, lab) for k, _, lab in CAT)[dk]
        short = dlab if dk != "clean" else "$A{+}B$"
        ax.text(xi, 1.04, f"{short}\n{hits[c]:.2f}", ha="center",
                fontsize=6.4, color=dcol if dk != "clean" else MUT)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(xt, fontsize=6.6)
    ax.set_ylim(0, 1.26)
    ax.set_yticks([0, 0.5, 1.0])
    ax.text(0.5, 1.03, mname, transform=ax.transAxes, ha="center",
            fontsize=7.2, style="italic")
    ax.grid(axis="y", color="#efefef", lw=0.6)
    ax.set_axisbelow(True)
    tag(ax, letter, x=-0.16 if letter == "c" else -0.10)
axQ1.set_ylabel("Fraction of answers")
axQ2.tick_params(labelleft=False)
handles = [Patch(fc=c, label=l) for _, c, l in CAT]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.4,
           bbox_to_anchor=(0.55, 0.005), handlelength=1.1,
           columnspacing=1.2, title="Final answer equals", title_fontsize=6.4)

save(fig, "F2_registers")
