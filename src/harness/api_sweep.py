"""Phase 1 behavioral sweep through APIs (Bedrock or Anthropic).

Same conditions and logging as the local runner, for models we cannot open.
Frontier-scale points anchor the cross-scale comparison. Concurrency kept
low to stay under throttling limits; every response cached to disk so
reruns resume.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import FAMILIES  # noqa: E402
from analysis.externalization import externalization_record  # noqa: E402
from harness.generate import (DIRECT_SUFFIX, COT_SUFFIX, extract_answer,  # noqa: E402
                              prompt_values)


def call_bedrock(client, model_id, text, max_tokens, temperature):
    for attempt in range(6):
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": text}]}],
                inferenceConfig={"maxTokens": max_tokens,
                                 "temperature": temperature})
            parts = resp["output"]["message"]["content"]
            reasoning = "".join(
                p.get("reasoningContent", {}).get("reasoningText", {})
                .get("text", "") for p in parts)
            answer = "".join(p.get("text", "") for p in parts)
            return reasoning, answer, resp["usage"]["outputTokens"]
        except Exception as e:
            if "Throttl" in str(e) or "TooMany" in str(e):
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("throttled out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--difficulties", default="1,2,4,8,16,32")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--conditions", default="direct,free")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import boto3
    client = boto3.client("bedrock-runtime", region_name=args.region)

    done = set()
    if os.path.exists(args.out):
        kept = []
        for line in open(args.out):
            r = json.loads(line)
            if "error" in r:
                continue  # errored items get retried, not marked done
            done.add((r["condition"], r["difficulty"], r["seed"]))
            kept.append(line)
        # rewrite without error rows so the file stays clean on resume
        with open(args.out, "w") as fo:
            fo.writelines(kept)

    gen = FAMILIES[args.family]
    jobs = []
    # conditions: direct, free, cot, or budget:N (think, but at most N
    # tokens; the cap is enforced by maxTokens so the model must choose
    # what to write down)
    for cond in args.conditions.split(","):
        for d in [int(x) for x in args.difficulties.split(",")]:
            for s in range(args.n):
                if (cond, d, s) not in done:
                    jobs.append((cond, d, s))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "a")

    def run(job):
        cond, d, s = job
        inst = gen(d, s)
        budget = int(cond.split(":")[1]) if cond.startswith("budget") else None
        if cond == "direct":
            text = inst.prompt + DIRECT_SUFFIX
            max_toks = 200
        elif budget is not None:
            text = (inst.prompt +
                    f"\nYou have a strict budget of {budget} tokens for "
                    "your entire response. Show only the working you most "
                    "need, then give the final answer as 'Answer: X'.")
            max_toks = budget
        else:
            text = inst.prompt + (COT_SUFFIX if cond == "cot" else "")
            max_toks = args.max_tokens
        temp = 0.0 if cond == "direct" else args.temperature
        try:
            reasoning, answer, out_toks = call_bedrock(
                client, args.model_id, text, max_toks, temp)
        except Exception as e:
            return {"model": args.model_id, "condition": cond,
                    "difficulty": d, "seed": s, "error": str(e)[:200]}
        full = (reasoning + "\n" + answer).strip()
        pred = extract_answer(answer or full, cond)
        ext = None
        if cond != "direct":
            ext = externalization_record(full, inst.intermediates,
                                         prompt_values(inst))
        return {"model": args.model_id, "condition": cond, "difficulty": d,
                "seed": s, "answer": inst.answer, "pred": pred,
                "correct": pred.strip().lower()
                    == inst.answer.strip().lower(),
                "trace_tokens": out_toks, "externalization": ext,
                "trace": full[:20000]}

    with ThreadPoolExecutor(args.concurrency) as ex:
        for i, rec in enumerate(ex.map(run, jobs)):
            f.write(json.dumps(rec) + "\n")
            if i % 20 == 0:
                f.flush()
                print(f"{i+1}/{len(jobs)}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
