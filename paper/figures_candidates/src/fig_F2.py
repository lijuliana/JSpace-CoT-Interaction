"""Figure F2: written positions are settable, compositional registers.

Panel A: schematic of the state patch (same visible text, different stored
value). Panel B: per-model rates with intervals. Panels C, D: answer
category by planted condition for the two-register composition test.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
import numpy as np
from analysis.figstyle import apply

apply()
BLUE, VERM, GREEN, SKY = "#0072B2", "#D55E00", "#009E73", "#56B4E9"
INK, MUT, GRAY = "#1a1a1a", "#666666", "#9a9a9a"
T_GRAY, T_GREEN, T_BLUE = "#f0f0ee", "#e4f4ec", "#e8f1f8"
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")


def load(name):
    with open(os.path.join(ROOT, "results", "raw", name)) as f:
        return [json.loads(l) for l in f]


def cats(fname):
    """Answer-category fractions per condition for one s1 file."""
    conds = ["none", "aonly", "bonly", "joint"]
    out = {c: {"clean": 0, "apB": 0, "Abp": 0, "apbp": 0, "other": 0, "n": 0}
           for c in conds}
    for r in load(fname):
        exp = {"clean": r["A"] + r["B"], "apB": r["ap"] + r["B"],
               "Abp": r["A"] + r["bp"], "apbp": r["ap"] + r["bp"]}
        if len(set(exp.values())) < 4:
            continue
        rev = {str(v): k for k, v in exp.items()}
        for c in conds:
            for a in r[c]["answers"]:
                out[c][rev.get(a, "other")] += 1
                out[c]["n"] += 1
    for c in conds:
        n = out[c].pop("n")
        for k in out[c]:
            out[c][k] /= n
    return out


def band(ax, x, y, w, h, text, fc, tc=INK, fs=6.4, mono=True):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, family="monospace" if mono else "sans-serif")


fig = plt.figure(figsize=(5.6, 4.3))
gsT = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.08,
                       left=0.03, right=0.99, top=0.94, bottom=0.60)
gsB = fig.add_gridspec(1, 2, wspace=0.14, left=0.10, right=0.99,
                       top=0.43, bottom=0.15)
axS = fig.add_subplot(gsT[0])
axD = fig.add_subplot(gsT[1])
axQ1, axQ2 = fig.add_subplot(gsB[0]), fig.add_subplot(gsB[1])

# ---- Panel A: schematic ----
axS.axis("off")
axS.set_xlim(0, 1)
axS.set_ylim(0, 1)
axS.set_title("same text, different stored value,\ndifferent answer", fontsize=6.8)
axS.text(0.5, 0.90, "visible text (unchanged)", ha="center", fontsize=5.8,
         color=MUT)
band(axS, 0.10, 0.72, 0.80, 0.13, "After step 5 the value is 457.", T_GRAY)
axS.text(0.5, 0.62, "hidden state at that token", ha="center", fontsize=5.8,
         color=MUT)
band(axS, 0.20, 0.42, 0.20, 0.13, "457", T_GRAY)
band(axS, 0.60, 0.42, 0.20, 0.13, "462", T_GREEN, tc=GREEN)
axS.add_patch(FancyArrowPatch((0.43, 0.485), (0.57, 0.485), arrowstyle="->",
                              color=GREEN, lw=1.3, mutation_scale=9))
band(axS, 0.06, 0.13, 0.88, 0.15, "continuation computes\nfrom 462, not 457",
     T_BLUE, tc=BLUE, fs=6.2, mono=False)

# ---- Panel B: per-model dot plot ----
models = ["Qwen3-4B", "Qwen2.5-7B", "Phi-3-medium", "OLMo-2-7B",
          "R1-distill-7B", "R1-distill-14B"]
restore = [(1.00, 1.00, 1.00), (0.97, 0.92, 1.00), (0.98, 0.95, 1.00),
           (0.88, 0.82, 0.94), (1.00, 0.99, 1.00), (0.99, 0.98, 1.00)]
plant = [(1.00, 0.99, 1.00), (0.74, 0.67, 0.81), (0.94, 0.91, 0.97),
         (0.93, 0.88, 0.97), (0.76, 0.68, 0.84), (0.70, 0.61, 0.80)]
rand = [(0.00, 0.00, 0.00), (0.00, 0.00, 0.01), (0.00, 0.00, 0.00),
        (0.00, 0.00, 0.00), (0.30, 0.22, 0.39), (0.21, 0.13, 0.29)]
y = np.arange(6)[::-1]
for data, color, label, o in [(plant, BLUE, "plant new value", 0.0),
                              (restore, GREEN, "restore correct", 0.24),
                              (rand, GRAY, "random control", -0.24)]:
    m = [d[0] for d in data]
    lo = [d[0] - d[1] for d in data]
    hi = [d[2] - d[0] for d in data]
    axD.errorbar(m, y + o, xerr=[lo, hi], fmt="o", color=color, ms=4.6,
                 mec="white", mew=0.9, elinewidth=1.0, label=label, zorder=3)
axD.set_yticks(y)
axD.set_yticklabels(models, fontsize=6.2)
axD.set_xlabel("answer follows patched state", fontsize=6.6)
axD.set_xlim(-0.04, 1.06)
axD.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axD.set_xticklabels(["0", "", "0.5", "", "1"])
axD.grid(axis="x", color="#ececec", lw=0.7)
axD.set_axisbelow(True)
axD.legend(fontsize=5.6, loc="upper left", bbox_to_anchor=(0.10, 1.02),
           frameon=False)

# ---- Panels C, D: composition answer categories ----
CAT = [("clean", GRAY, "clean A+B"), ("apB", VERM, "a′+B"),
       ("Abp", GREEN, "A+b′"), ("apbp", BLUE, "a′+b′"),
       ("other", "#dcdcdc", "other")]
DOM = {"none": "clean", "aonly": "apB", "bonly": "Abp", "joint": "apbp"}
conds = ["none", "aonly", "bonly", "joint"]
xt = ["no\npatch", "plant\nA", "plant\nB", "plant\nboth"]
for ax, fname, title in [(axQ1, "s1_qwen25-7b.jsonl", "Qwen2.5-7B"),
                         (axQ2, "s1_qwen3-4b.jsonl", "Qwen3-4B")]:
    data = cats(fname)
    x = np.arange(4)
    for xi, c in enumerate(conds):
        bot = 0.0
        for key, color, _ in CAT:
            v = data[c][key]
            if v > 0:
                ax.bar(xi, v, 0.58, bottom=bot, color=color,
                       edgecolor="white", linewidth=1.0)
            bot += v
        dk = DOM[c]
        dcol = dict((k, col) for k, col, _ in CAT)[dk]
        dlab = dict((k, lab) for k, _, lab in CAT)[dk]
        ax.text(xi, 1.03, f"{dlab}\n{data[c][dk]:.2f}", ha="center",
                fontsize=5.8, color=dcol if dk != "clean" else MUT)
    ax.set_xticks(x)
    ax.set_xticklabels(xt, fontsize=6.2)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title(title, fontsize=6.8, pad=10)
    ax.grid(axis="y", color="#ececec", lw=0.7)
    ax.set_axisbelow(True)
axQ1.set_ylabel("fraction of answers")
axQ2.tick_params(labelleft=False)
fig.text(0.545, 0.485, "the answer category tracks which registers were set",
         ha="center", fontsize=6.8)
handles = [Patch(fc=c, label=l) for _, c, l in CAT]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=5.8,
           frameon=False, bbox_to_anchor=(0.545, 0.005),
           handlelength=1.1, columnspacing=1.2)

out = os.path.join(os.path.dirname(__file__), "..")
fig.savefig(os.path.join(out, "F2_registers.pdf"))
fig.savefig("/tmp/cur_F2_registers.png", dpi=170)
print("F2 done")
