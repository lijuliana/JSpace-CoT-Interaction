"""R4: capacity-boundary sweep. Deep rendering only; the manipulated
variable is the edit position j, which in the deep rendering equals the
rederivation depth delta of the edited value. Follow-versus-delta curves
per model; the crossover locates the model's boundary, to be compared with
its independently measured closed-channel capacity r(m).

API models here; white-box models run the same protocol via r1_recompute
conventions once the GPU frees up.
"""

import argparse
import json
import random
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.api_readback import converse  # noqa: E402
from harness.r1_api import llama_invoke  # noqa: E402
from harness.r1_recompute import (  # noqa: E402
    make_item, forward_from, prefix, final_answer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--js", type=int, nargs="+",
                    default=[2, 4, 8, 12, 16, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import boto3
    client = boto3.client("bedrock-runtime", region_name=args.region)
    fout = open(args.out, "a")

    for seed in args.seeds:
        rng = random.Random(4000 + seed)
        jobs = []
        for j in args.js:
            for ix in range(args.n):
                it = make_item(rng, args.d, j)
                delta = rng.choice([x for x in range(-9, 10) if x != 0])
                vj_bad = it["vals"][j] + delta
                for cell, shown in (("edit", vj_bad),
                                    ("floor", it["vals"][j])):
                    jobs.append((ix, j, cell, it["deep_prompt"],
                                 prefix(it, shown),
                                 forward_from(it, j, vj_bad),
                                 str(it["answer"])))

        def run(job):
            ix, j, cell, p, pre, tgt, clean = job
            if "anthropic" in args.model_id:
                pre = pre.rstrip()
            try:
                if "meta.llama" in args.model_id:
                    text = llama_invoke(client, args.model_id, p, pre,
                                        args.max_tokens)
                else:
                    text = converse(client, args.model_id, p, pre,
                                    args.max_tokens)
            except Exception as e:
                return {"exp": "r4", "model": args.model_id, "seed": seed,
                        "item": ix, "j": j, "cell": cell,
                        "error": str(e)[:200]}
            ans = final_answer(text)
            return {"exp": "r4", "model": args.model_id, "seed": seed,
                    "item": ix, "d": args.d, "j": j, "cell": cell,
                    "answer": ans, "clean": clean, "edit_target": str(tgt),
                    "follows_edit": ans == str(tgt),
                    "follows_clean": ans == clean}

        with ThreadPoolExecutor(args.concurrency) as ex:
            for i, rec in enumerate(ex.map(run, jobs)):
                fout.write(json.dumps(rec) + "\n")
                if i % 200 == 0:
                    fout.flush()
                    print(f"seed {seed}: {i}/{len(jobs)}", flush=True)
        fout.flush()
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
