"""Phase 1 analysis: accuracy, trace length, and externalization curves vs
difficulty, with bootstrap CIs over instances. Reads the jsonl the harness
writes, emits a summary csv and plots."""

import argparse
import json
from collections import defaultdict

import numpy as np


def bootstrap_ci(vals, n_boot=2000, seed=0):
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, (n_boot, len(vals))).mean(axis=1)
    return (vals.mean(), *np.percentile(means, [2.5, 97.5]))


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if "error" in r:
                    continue
                r.pop("trace", None)
                rows.append(r)
    return rows


def summarize(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["condition"], r["difficulty"])].append(r)
    out = []
    for (model, cond, d), rs in sorted(cells.items()):
        acc = bootstrap_ci([r["correct"] for r in rs])
        toks = bootstrap_ci([r["trace_tokens"] for r in rs])
        ext = bootstrap_ci([
            (r.get("externalization") or {}).get("externalization_fraction")
            for r in rs])
        out.append({
            "model": model.split("/")[-1], "condition": cond,
            "difficulty": d, "n": len(rs),
            "acc": acc[0], "acc_lo": acc[1], "acc_hi": acc[2],
            "tokens": toks[0], "tokens_lo": toks[1], "tokens_hi": toks[2],
            "ext_frac": ext[0], "ext_lo": ext[1], "ext_hi": ext[2],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--plot")
    args = ap.parse_args()

    rows = load(args.inputs)
    summary = summarize(rows)

    import csv as csvmod
    with open(args.csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {args.csv} ({len(summary)} cells)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conds = sorted({s["condition"] for s in summary})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for cond in conds:
            ss = [s for s in summary if s["condition"] == cond]
            ds = [s["difficulty"] for s in ss]
            for ax, key, lo, hi, label in [
                    (axes[0], "acc", "acc_lo", "acc_hi", "accuracy"),
                    (axes[1], "tokens", "tokens_lo", "tokens_hi",
                     "trace tokens"),
                    (axes[2], "ext_frac", "ext_lo", "ext_hi",
                     "externalization fraction")]:
                ys = [s[key] for s in ss]
                ax.plot(ds, ys, marker="o", label=cond)
                ax.fill_between(ds, [s[lo] for s in ss],
                                [s[hi] for s in ss], alpha=0.2)
                ax.set_xlabel("difficulty")
                ax.set_ylabel(label)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
