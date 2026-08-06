"""Read out the eviction probe: decodability of the target value vs steps
elapsed since its computation, written vs suppressed variant, at the best
layer. Eviction = written decodability falls below suppressed at matched
elapsed; redundant cache = written stays at or above suppressed."""

import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--csv")
    args = ap.parse_args()

    rows = []
    for p in args.inputs:
        rows += [json.loads(l) for l in open(p)]

    # pick the layer with the highest mean target r2 at elapsed 0 (where the
    # value is freshly computed and should be maximally present)
    by_layer = defaultdict(list)
    for r in rows:
        if r["elapsed"] == 0:
            by_layer[r["layer"]].append(r["r2"])
    best = max(by_layer, key=lambda l: sum(by_layer[l]) / len(by_layer[l]))
    print(f"best layer (peak fresh decodability): {best}\n")

    cells = defaultdict(dict)
    for r in rows:
        if r["layer"] != best:
            continue
        cells[r["elapsed"]][r["variant"]] = r

    print(f"{'elapsed':>7} {'written':>8} {'suppress':>8} {'w-ctrl':>7} "
          f"{'s-ctrl':>7} {'gap(w-s)':>9}")
    out = []
    for e in sorted(cells):
        w = cells[e].get("written", {})
        s = cells[e].get("suppressed", {})
        if not w or not s:
            continue
        gap = w["r2"] - s["r2"]
        print(f"{e:>7} {w['r2']:>8.3f} {s['r2']:>8.3f} "
              f"{w['r2_control']:>7.3f} {s['r2_control']:>7.3f} {gap:>9.3f}")
        out.append({"elapsed": e, "layer": best, "written_r2": w["r2"],
                    "suppressed_r2": s["r2"], "gap": gap,
                    "written_control": w["r2_control"],
                    "suppressed_control": s["r2_control"]})

    if out:
        early = [o["gap"] for o in out if o["elapsed"] in (1, 2, 3)]
        if early:
            m = sum(early) / len(early)
            print(f"\nmean written-minus-suppressed gap at elapsed 1-3: "
                  f"{m:+.3f}")
            print("negative => eviction (wrote it, then let it decay); "
                  "positive => redundant cache (kept the written value too)")

    if args.csv and out:
        import csv as csvmod
        with open(args.csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
