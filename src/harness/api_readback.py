"""Frontier-scale behavioral read-back via API assistant prefill.

We cannot patch residuals through an API, but we can run the behavioral
half of the read-back test at frontier scale: ask the model to solve a
variable chain writing each value, corrupt one written value, prefill the
assistant turn with the corrupted partial trace, and let the model continue.
If the answer follows the corruption, the model read its own written value
rather than recomputing from the operands.

Uses Bedrock Converse assistant prefill (the last message is an assistant
turn the model continues). Non-reasoning models are cleaner here because a
reasoning model re-solves after its think block; we use V3.2 and Claude.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import variable_chain  # noqa: E402

SOLVE_INSTR = ("\nSolve it by writing one line per step in the form "
               "'name = prev op arg = value', computing each value, then a "
               "final line 'Answer: X'. Write only these lines.")


def converse(client, model_id, user_text, assistant_prefill, max_tokens):
    msgs = [{"role": "user", "content": [{"text": user_text}]}]
    if assistant_prefill is not None:
        msgs.append({"role": "assistant",
                     "content": [{"text": assistant_prefill}]})
    for attempt in range(6):
        try:
            r = client.converse(
                modelId=model_id, messages=msgs,
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.6})
            return "".join(p.get("text", "")
                           for p in r["output"]["message"]["content"])
        except Exception as e:
            if "Throttl" in str(e) or "TooMany" in str(e):
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("throttled out")


def extract_answer(text):
    m = re.findall(r"[Aa]nswer\s*[:=]?\s*(-?\d+)", text)
    return m[-1] if m else ""


def forward_from(inst, idx, val):
    v = val
    for op, arg in inst.meta["ops"][idx:]:
        v = v + arg if op == "+" else v - arg
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--delta", type=int, default=40)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import boto3
    client = boto3.client("bedrock-runtime", region_name=args.region)

    def run(s):
        inst = variable_chain(args.depth, 70_000 + s)
        # 1. get a clean solved trace
        base = converse(client, args.model_id, inst.prompt + SOLVE_INSTR,
                        None, 1200)
        if extract_answer(base) != inst.answer:
            return None  # only corrupt correct solves
        # 2. find the target mid-chain value on its own line
        tgt = len(inst.meta["ops"]) // 2
        name = inst.intermediates[tgt + 1][0]
        val = int(inst.intermediates[tgt + 1][1])
        corr = val + args.delta
        clean_ans = forward_from(inst, tgt + 1, val)
        corr_ans = forward_from(inst, tgt + 1, corr)
        if clean_ans == corr_ans:
            return None
        # locate the line that assigns `name` and ends in the value
        lines = base.splitlines()
        cut, ok = [], False
        for ln in lines:
            m = re.match(rf"\s*{re.escape(name)}\s*=.*=\s*(-?\d+)\s*$", ln)
            if m and int(m.group(1)) == val:
                cut.append(re.sub(r"(=\s*)-?\d+(\s*)$",
                                  rf"\g<1>{corr}\g<2>", ln))
                ok = True
                break
            cut.append(ln)
        if not ok:
            return None
        prefill = "\n".join(cut)
        # 3. continue from the corrupted prefill
        cont = converse(client, args.model_id, inst.prompt + SOLVE_INSTR,
                        prefill, 1200)
        pred = extract_answer(cont) or extract_answer(prefill + cont)
        return {"model": args.model_id, "depth": args.depth, "seed": s,
                "clean_val": val, "corr_val": corr,
                "clean_ans": clean_ans, "corr_ans": corr_ans, "pred": pred,
                "follows_corruption": pred == corr_ans,
                "follows_clean": pred == clean_ans}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = []
    with ThreadPoolExecutor(args.concurrency) as ex:
        for rec in ex.map(run, range(args.n)):
            if rec:
                kept.append(rec)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    import numpy as np
    if kept:
        fc = np.mean([r["follows_corruption"] for r in kept])
        cl = np.mean([r["follows_clean"] for r in kept])
        print(f"{args.model_id}: n={len(kept)} "
              f"follows_corruption={fc:.2f} follows_clean={cl:.2f}")


if __name__ == "__main__":
    main()
