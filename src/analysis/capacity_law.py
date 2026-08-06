"""Internal serial capacity across models.

The reframed onset quantity: not where externalization begins (it is
saturated from d=1) but how deep a serial computation the model completes
in a single forward pass with no writing. We estimate d_int, the largest
difficulty at which direct-answer accuracy stays above a threshold, per
model, and relate it to depth and parameter count.

This is the honest version of the onset law. Free-generation externalization
gave no crossover, so the measurable capacity boundary is the direct cliff,
and its cross-model variance is what a law can be fit to.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# layer counts for depth axis
DEPTH = {
    "1.5b": 28, "7b": 28, "14b": 48,
    "llama8b": 32, "llama70b": 80,
    "v32": 61, "r1_671b": 61,
}
PARAMS_B = {
    "1.5b": 1.5, "7b": 7, "14b": 14,
    "llama8b": 8, "llama70b": 70, "v32": 671, "r1_671b": 671,
}


def d_int(direct_rows, thresh=0.5):
    """largest d with direct accuracy >= thresh; interpolate between the
    last-above and first-below points for a continuous estimate."""
    by_d = defaultdict(list)
    for r in direct_rows:
        by_d[r["difficulty"]].append(r["correct"])
    xs = sorted(by_d)
    acc = {d: np.mean(by_d[d]) for d in xs}
    above = [d for d in xs if acc[d] >= thresh]
    if not above:
        return 0.0
    last = max(above)
    nxt = [d for d in xs if d > last]
    if not nxt:
        return float(last)
    first_below = min(nxt)
    # linear interpolation on accuracy between last and first_below
    a0, a1 = acc[last], acc[first_below]
    if a0 == a1:
        return float(last)
    frac = (a0 - thresh) / (a0 - a1)
    return last + frac * (first_below - last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--family", default="var")
    args = ap.parse_args()

    tags = {
        "1.5b": f"p1_{args.family}_1.5b.jsonl",
        "7b": f"p1_{args.family}_7b.jsonl",
        "14b": f"p1_{args.family}_14b.jsonl",
        "llama8b": f"p1_{args.family}_llama8b.jsonl",
        "llama70b": f"p1_{args.family}_llama70b.jsonl",
        "r1_671b": f"p1_{args.family}_r1_671b.jsonl",
    }
    # reasoning models emit think tokens even when asked to answer directly
    # (API cannot force-suppress), so the direct cliff is invalid for them
    REASONING = {"r1_671b", "1.5b", "7b", "14b"}
    print(f"family={args.family}, d_int = largest d with direct acc >= 0.5")
    print("(non-reasoning models only for the fit; distills shown but "
          "flagged, their no-think direct is off-distribution)\n")
    print(f"{'model':>10} {'layers':>7} {'params':>7} {'d_int':>6} {'fit':>4}")
    pts = []
    for tag, fname in tags.items():
        path = os.path.join(args.raw, fname)
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        direct = [r for r in rows if r.get("condition") == "direct"
                  and "error" not in r]
        if not direct:
            continue
        di = d_int(direct)
        in_fit = tag not in REASONING
        pts.append((tag, DEPTH[tag], PARAMS_B[tag], di, in_fit))
        print(f"{tag:>10} {DEPTH[tag]:>7} {PARAMS_B[tag]:>7} {di:>6.2f} "
              f"{'y' if in_fit else 'n':>4}")

    fit = [p for p in pts if p[4]]
    if len(fit) >= 3:
        depth = np.array([p[1] for p in fit], float)
        logp = np.log([p[2] for p in fit])
        di = np.array([p[3] for p in fit], float)
        for name, x in (("depth", depth), ("log-params", logp)):
            if np.std(x) < 1e-9:
                continue
            r = np.corrcoef(x, di)[0, 1]
            print(f"\ncorr(d_int, {name}) = {r:+.2f}")
        print("\nnote: d_int is small (1-8) and clipped by the difficulty "
              "grid; treat as ordinal evidence of cross-model capacity "
              "variance, not a precise fit. Llama-70b is the outlier with "
              "d_int ~ 4-8; distills and the deepseek pair sit at 1-2.")


if __name__ == "__main__":
    main()
