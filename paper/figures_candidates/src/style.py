"""Shared style for all candidate paper figures.

Serif text matching the NeurIPS body font (Times), STIX math, no in-plot
titles (takeaways live in captions), sentence-case labels, panel letters.

Color meanings, constant across figures:
  BLUE      the written trace channel and the state patch (planting a value)
  VERM      internal-workspace interventions and the silent condition
  SKY       attention knockout
  GREEN     restoring or matching the correct state
  GRAY      random and matched controls
Model identity, when lines are per-model, uses MODEL_COLORS.
Grayscale safety comes from distinct markers and line styles per series.
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")

BLUE = "#31518F"
VERM = "#C05B33"
SKY = "#5D9FDB"
GREEN = "#4F8D57"
GRAY = "#8a8a8a"
INK = "#1a1a1a"
MUT = "#666666"
LIGHT = "#dcdcdc"

T_GRAY, T_VERM, T_GREEN, T_BLUE, T_SKY = (
    "#f0f0ee", "#f8eae2", "#e9f2ea", "#e9eef7", "#eaf1fa")

MODEL_COLORS = {
    "Qwen3-4B": SKY,
    "Qwen2.5-7B": BLUE,
    "Phi-3-medium": VERM,
    "OLMo-2-7B": GREEN,
}
MODEL_MARKERS = {
    "Qwen3-4B": "o",
    "Qwen2.5-7B": "s",
    "Phi-3-medium": "^",
    "OLMo-2-7B": "D",
}


def apply():
    matplotlib.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIX Two Text",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": "#444444",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "axes.labelcolor": INK,
        "text.color": INK,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def tag(ax, letter, x=-0.14, y=1.02):
    """Bold uppercase panel letter in the axes corner."""
    ax.text(x, y, letter.upper(), transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(name):
    import json
    with open(os.path.join(ROOT, "results", "raw", name)) as f:
        return [json.loads(l) for l in f]


def save(fig, stem):
    fig.savefig(os.path.join(OUT, stem + ".pdf"))
    fig.savefig("/tmp/cur_" + stem + ".png", dpi=170)
    print(stem, "done")
