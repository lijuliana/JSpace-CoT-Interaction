"""R1 API arm: within-task recomputability manipulation on frontier models.

Same items and cells as r1_recompute.py (deep vs shallow rendering, edit vs
floor), run through Bedrock converse with assistant prefill instead of local
generation. This is the arm where the within-model gap is most expected:
V3.2 already recomputes depth-1 GSM8K intermediates (0.10 follow) while
following deep-chain edits (0.78), so the same contrast within one item set
is the direct test.
"""

import argparse
import json
import random
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.api_readback import converse  # noqa: E402
from harness.r1_recompute import (  # noqa: E402
    make_item, forward_from, prefix, final_answer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import boto3
    client = boto3.client("bedrock-runtime", region_name=args.region)
    fout = open(args.out, "a")

    for seed in args.seeds:
        rng = random.Random(1000 + seed)
        items = [make_item(rng, args.d, rng.randint(4, 7))
                 for _ in range(args.n)]
        jobs = []
        for ix, it in enumerate(items):
            delta = rng.choice([x for x in range(-9, 10) if x != 0])
            vj_bad = it["vals"][it["j"]] + delta
            for rend in ("deep", "shallow"):
                p = it[f"{rend}_prompt"]
                jobs.append((ix, rend, "edit", p, prefix(it, vj_bad),
                             forward_from(it, it["j"], vj_bad)))
                jobs.append((ix, rend, "floor", p,
                             prefix(it, it["vals"][it["j"]]),
                             forward_from(it, it["j"], vj_bad)))

        def run(job):
            ix, rend, cell, p, pre, tgt = job
            it = items[ix]
            try:
                text = converse(client, args.model_id, p, pre,
                                args.max_tokens)
            except Exception as e:
                return {"exp": "r1api", "model": args.model_id,
                        "seed": seed, "item": ix, "rendering": rend,
                        "cell": cell, "error": str(e)[:200]}
            ans = final_answer(text)
            return {"exp": "r1api", "model": args.model_id, "seed": seed,
                    "item": ix, "d": args.d, "j": it["j"],
                    "rendering": rend, "cell": cell, "answer": ans,
                    "clean": str(it["answer"]), "edit_target": str(tgt),
                    "follows_edit": ans == str(tgt),
                    "follows_clean": ans == str(it["answer"])}

        with ThreadPoolExecutor(args.concurrency) as ex:
            for i, rec in enumerate(ex.map(run, jobs)):
                fout.write(json.dumps(rec) + "\n")
                if i % 100 == 0:
                    fout.flush()
                    print(f"seed {seed}: {i}/{len(jobs)}", flush=True)
        fout.flush()
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
