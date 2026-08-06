"""Rescore stored traces after an extract_answer fix. Rewrites pred and
correct in place (backing up the original). Applies to any harness jsonl
that stores the trace."""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.generate import extract_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    for path in args.inputs:
        shutil.copy(path, path + ".bak")
        rows = [json.loads(l) for l in open(path)]
        changed = 0
        for r in rows:
            if "error" in r or "trace" not in r:
                continue
            pred = extract_answer(r["trace"], r.get("condition", "free"))
            correct = pred.strip().lower() == r["answer"].strip().lower()
            if pred != r.get("pred") or correct != r.get("correct"):
                changed += 1
            r["pred"], r["correct"] = pred, correct
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{path}: {changed} rows changed")


if __name__ == "__main__":
    main()
