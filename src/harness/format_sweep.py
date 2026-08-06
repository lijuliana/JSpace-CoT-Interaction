"""Phase 5: external representation geometry.

Same variable-chain instances, different scratchpad formats requested by
instruction. If written tokens are a memory store, the format that makes
values easiest to address should give more accuracy per written token and
tolerate deeper chains at a fixed budget. Formats:

  prose      - natural language working, the model's default
  state      - a running state line after each step (an explicit register
               dump: "state: a=347 b=399 c=366 ...")
  code       - code-like assignments, one per line, no prose

The comparison is accuracy and accuracy-per-token across formats at matched
difficulty, plus externalization fraction (should stay ~1 in all; the
question is efficiency, not whether values are written).

Runs through the API (no white-box needed).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import variable_chain  # noqa: E402
from analysis.externalization import externalization_record  # noqa: E402
from harness.generate import extract_answer, prompt_values  # noqa: E402
from harness.api_sweep import call_bedrock  # noqa: E402

FORMATS = {
    "prose": "Work through it step by step in prose, then give the final "
             "answer as 'Answer: X'.",
    "state": "After each step, write a running state line listing the "
             "current value of every variable so far, in the form "
             "'state: a=.. b=.. c=..'. Then give the final answer as "
             "'Answer: X'.",
    "code": "Write the solution as a list of code-like assignment lines, one "
            "per step, no prose. Then give the final answer as 'Answer: X'.",
    "code_eval": "Write the solution as code-like assignment lines, one per "
                 "step, and after each assignment write its evaluated numeric "
                 "value as a comment, like 'b = a + 52  # 399'. No prose. "
                 "Then give the final answer as 'Answer: X'.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--difficulties", default="8,16,32,48")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import boto3
    client = boto3.client("bedrock-runtime", region_name=args.region)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            if "error" not in r:
                done.add((r["format"], r["difficulty"], r["seed"]))

    jobs = []
    for fmt in FORMATS:
        for d in [int(x) for x in args.difficulties.split(",")]:
            for s in range(args.n):
                if (fmt, d, s) not in done:
                    jobs.append((fmt, d, s))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "a")

    def run(job):
        fmt, d, s = job
        inst = variable_chain(d, s)
        text = inst.prompt + "\n" + FORMATS[fmt]
        try:
            reasoning, answer, out_toks = call_bedrock(
                client, args.model_id, text, args.max_tokens,
                args.temperature)
        except Exception as e:
            return {"format": fmt, "difficulty": d, "seed": s,
                    "error": str(e)[:200]}
        full = (reasoning + "\n" + answer).strip()
        pred = extract_answer(answer or full, "cot")
        ext = externalization_record(full, inst.intermediates,
                                     prompt_values(inst))
        return {"model": args.model_id, "format": fmt, "difficulty": d,
                "seed": s, "answer": inst.answer, "pred": pred,
                "correct": pred.strip().lower() == inst.answer.strip().lower(),
                "trace_tokens": out_toks,
                "ext_frac": ext["externalization_fraction"],
                "trace": full[:12000]}

    with ThreadPoolExecutor(args.concurrency) as ex:
        for i, rec in enumerate(ex.map(run, jobs)):
            f.write(json.dumps(rec) + "\n")
            if i % 20 == 0:
                f.flush()
                print(f"{i+1}/{len(jobs)}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
