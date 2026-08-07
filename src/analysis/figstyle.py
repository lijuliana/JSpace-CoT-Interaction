"""Shared matplotlib style for all paper figures.

Column width for NeurIPS is 5.5 in single-column figures in a one-column
layout; we target 3.3 in for half-width and 5.5 in for full-width. Fonts
sized to remain readable at print size. Colorblind-safe palette (Okabe-Ito).
"""

import matplotlib

OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermilion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}

MODEL_COLORS = {
    "Qwen2.5-7B": OKABE_ITO["blue"],
    "Qwen3-4B": OKABE_ITO["sky"],
    "Phi-3-medium": OKABE_ITO["vermilion"],
    "OLMo-2-7B": OKABE_ITO["green"],
    "R1-distill-7B": OKABE_ITO["purple"],
    "R1-distill-14B": OKABE_ITO["orange"],
    "Llama-3.3-70B": OKABE_ITO["yellow"],
    "DeepSeek V3.2": OKABE_ITO["black"],
    "Claude Sonnet 4.5": OKABE_ITO["purple"],
}


def apply():
    matplotlib.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def bootstrap_ci(values, iters=10000, seed=0):
    """95 percent bootstrap CI for a mean of 0/1 or fractional values."""
    import random
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])
