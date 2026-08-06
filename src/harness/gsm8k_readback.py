"""Fix C: read-back on a real benchmark (GSM8K), behavioral, via API prefill.

All our tasks are synthetic. This checks the read-back generalizes to real
word problems. Procedure: have the model solve a GSM8K problem showing each
computed value, corrupt one mid-trace computed value that is not in the
question (so it is a genuine intermediate, not a given), continue from the
edit via assistant prefill, and measure whether the final answer changes
from the clean answer. Control: a no-corruption resample, to measure how
often the answer changes from sampling alone (the noise floor). A corrupt-
follow rate well above the noise floor means written intermediates are read
back on real problems too.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.api_readback import converse  # noqa: E402

SOLVE = ("\nSolve step by step. Show each calculation on its own line ending "
         "with '= <value>'. Then give the final answer on the last line as "
         "'#### <number>'.")


def final_answer(text):
    m = re.findall(r"####\s*\$?(-?[\d,]+)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else ""


def question_numbers(q):
    return set(re.findall(r"\d+", q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--data", default="data/gsm8k_test.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-steps", type=int, default=0,
                    help="only use problems whose gold answer has at least "
                    "this many <<>> calculation steps (depth filter)")
    ap.add_argument("--delta", type=int, default=7)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import boto3
    client = boto3.client("bedrock-runtime", region_name=args.region)
    allp = [json.loads(l) for l in open(args.data)]
    if args.min_steps:
        allp = [p for p in allp
                if p["answer"].count("<<") >= args.min_steps]
    problems = allp[:args.n]
    print(f"using {len(problems)} problems (min_steps={args.min_steps})",
          flush=True)

    def run(p):
        q = p["question"]
        gold = final_answer(p["answer"])
        base = converse(client, args.model_id, q + SOLVE, None, 800)
        clean_ans = final_answer(base)
        if clean_ans != gold:
            return None  # only edit correct solves
        qnums = question_numbers(q)
        # lines with a computed value '= N' where N is not a given number
        body = re.split(r"####", base)[0]
        lines = body.splitlines()
        cand = []
        for i, ln in enumerate(lines):
            m = re.search(r"=\s*\$?(-?\d+)\s*$", ln.strip())
            if m and m.group(1) not in qnums and i < len(lines) - 1:
                cand.append((i, m.group(1)))
        if not cand:
            return None
        # pick a mid-trace computed value
        idx, val = cand[len(cand) // 2]
        corr = str(int(val) + args.delta)
        cut = lines[:idx] + [re.sub(r"(=\s*\$?)-?\d+(\s*)$",
                                    rf"\g<1>{corr}\g<2>", lines[idx])]
        prefill = "\n".join(cut)
        cont = converse(client, args.model_id, q + SOLVE, prefill, 800)
        corr_ans = final_answer(prefill + "\n" + cont)
        # no-corruption resample control from the same clean prefix
        clean_prefix = "\n".join(lines[:idx + 1])
        cont2 = converse(client, args.model_id, q + SOLVE, clean_prefix, 800)
        resample_ans = final_answer(clean_prefix + "\n" + cont2)
        return {"gold": gold, "clean_ans": clean_ans,
                "corr_val": corr, "orig_val": val,
                "corr_ans": corr_ans, "resample_ans": resample_ans,
                "answer_changed": corr_ans != clean_ans and corr_ans != "",
                "resample_changed": resample_ans != clean_ans
                and resample_ans != ""}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = []
    with ThreadPoolExecutor(args.concurrency) as ex:
        for r in ex.map(run, problems):
            if r:
                kept.append(r)
    with open(args.out, "w") as fo:
        for r in kept:
            fo.write(json.dumps(r) + "\n")
    import numpy as np
    if kept:
        ch = np.mean([r["answer_changed"] for r in kept])
        fl = np.mean([r["resample_changed"] for r in kept])
        print(f"{args.model_id}: n={len(kept)} "
              f"answer_changed_under_corruption={ch:.2f} "
              f"resample_noise_floor={fl:.2f}")


if __name__ == "__main__":
    main()
