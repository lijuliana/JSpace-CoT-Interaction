"""Read out the format sweep: accuracy, tokens, accuracy-per-100-tokens, and
externalization by format and difficulty. The key contrast is code (symbolic,
values not evaluated) vs code_eval (same style, values written): if the only
difference that matters is whether values are externalized, code fails where
code_eval succeeds at matched style and difficulty."""

import argparse
import json
from collections import defaultdict

import numpy as np


def ci(v, n_boot=2000, seed=0):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n_boot, len(v))).mean(1)
    return (v.mean(), *np.percentile(m, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = []
    for p in args.inputs:
        rows += [json.loads(l) for l in open(p) if '"error"' not in l]

    cells = defaultdict(list)
    for r in rows:
        cells[(r["format"], r["difficulty"])].append(r)

    fmts = ["prose", "state", "code_eval", "code"]
    diffs = sorted({d for _, d in cells})
    out = []
    print(f"{'format':>10} {'d':>4} {'acc':>6} {'toks':>6} "
          f"{'acc/100tok':>10} {'ext':>6} n")
    for fmt in fmts:
        for d in diffs:
            rs = cells.get((fmt, d), [])
            if not rs:
                continue
            acc = np.mean([r["correct"] for r in rs])
            toks = np.mean([r["trace_tokens"] for r in rs])
            ext = ci([r.get("ext_frac") for r in rs])[0]
            eff = 100 * acc / toks if toks else 0
            out.append({"format": fmt, "difficulty": d, "acc": acc,
                        "tokens": toks, "acc_per_100tok": eff, "ext": ext,
                        "n": len(rs)})
            print(f"{fmt:>10} {d:>4} {acc:>6.2f} {toks:>6.0f} "
                  f"{eff:>10.3f} {ext:>6.2f} {len(rs)}")
        print()

    if args.csv and out:
        import csv as csvmod
        with open(args.csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
