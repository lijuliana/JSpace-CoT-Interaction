"""De-circularize the necessity result.

The concern: on variable chains a correct trace mechanically contains the
values because the last step reads 'x = y op z = answer' and y is the
previous value, so surface externalization = 1.0 is partly forced by format,
not memory. Two checks here:

1. Position decomposition: split externalization fraction among correct
   traces by intermediate position (early / middle / late). If the ceiling
   is format-forced it should be carried by the last one or two steps; if
   early intermediates are also always written, that is not format forcing.

2. dag_reachability necessity: on the DAG task the answer is a node label
   and the hop nodes are not accumulated into the answer, so writing them is
   not forced by the answer format. Compare externalization among correct vs
   wrong traces there.
"""

import argparse
import json
import sys
import os
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import variable_chain, dag_reachability  # noqa: E402
from analysis.externalization import (externalization_record,  # noqa: E402
                                      find_value_mentions)


def position_decomp(path, gen, depths):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("condition") == "free"
            and r["correct"] and r["difficulty"] in depths]
    buckets = defaultdict(lambda: {"early": [], "mid": [], "late": []})
    for r in rows:
        inst = gen(r["difficulty"], r["seed"])
        inters = inst.intermediates
        k = len(inters)
        pv = set()
        for i, (name, val) in enumerate(inters):
            frac_pos = i / max(1, k - 1)
            bucket = ("early" if frac_pos < 0.34 else
                      "mid" if frac_pos < 0.67 else "late")
            written = bool(find_value_mentions(r["trace"], val))
            buckets[r["difficulty"]][bucket].append(written)
    print("externalization among CORRECT traces, by position in chain:")
    print(f"{'d':>4} {'early':>8} {'mid':>8} {'late':>8}")
    for d in sorted(buckets):
        b = buckets[d]
        row = " ".join(
            f"{np.mean(b[k]):8.2f}" if b[k] else f"{'-':>8}"
            for k in ("early", "mid", "late"))
        print(f"{d:>4} {row}")
    print("if the late column is ~1.0 but early is well below, the ceiling "
          "was format-forced; if early is also ~1.0 it is not.\n")


def dag_necessity(path):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("condition") == "free"]
    by = defaultdict(lambda: {"ok": [], "bad": []})
    for r in rows:
        inst = dag_reachability(r["difficulty"], r["seed"])
        # hop nodes are the intermediates; the final one is the answer, so
        # exclude it (writing the answer is trivially forced)
        hops = inst.intermediates[:-1]
        if not hops:
            continue
        written = [bool(find_value_mentions(r["trace"], v)) for _, v in hops]
        frac = np.mean(written)
        by[r["difficulty"]]["ok" if r["correct"] else "bad"].append(frac)
    print("dag_reachability: externalization of intermediate hop nodes "
          "(answer excluded, so not format-forced), correct vs wrong:")
    print(f"{'d':>4} {'ext|correct':>12} {'ext|wrong':>12}")
    for d in sorted(by):
        ok, bad = by[d]["ok"], by[d]["bad"]
        o = f"{np.mean(ok):.2f} (n={len(ok)})" if ok else "-"
        b = f"{np.mean(bad):.2f} (n={len(bad)})" if bad else "-"
        print(f"{d:>4} {o:>12} {b:>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var")
    ap.add_argument("--dag")
    ap.add_argument("--depths", default="16,32,48")
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]
    if args.var:
        position_decomp(args.var, variable_chain, depths)
    if args.dag:
        dag_necessity(args.dag)


if __name__ == "__main__":
    main()
