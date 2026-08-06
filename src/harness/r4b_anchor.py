"""R4b: anchor-distance sweep. The R4 depth sweep showed follow rates are
not a smooth function of implicit rederivation depth (Sonnet follows 0.98
at j=1 but reverts at j=2-4; V3.2 follows 0.85 at j=1) while R1's shallow
rendering, which states the value's composition in the prompt, produces
strong reversion. Hypothesis: verification triggers on a salient
prompt-stated anchor, and succeeds only if the model can bridge the
distance from anchor to edited value internally.

Design v2 (after adversarial review): the v1 anchor was redundant with
the visible trace, so flat curves were overdetermined. Now the prefill is
truncated to start at step k+1 (k = j - delta), so the trace never shows
v_k; the anchor sentence is the sole source of that value. Cells cross
anchor presence with the same truncation: anchored vs unanchored at each
delta, edit vs floor. Rederivation of the edited value needs delta ops
from the anchor when present, or j ops from the prompt start when not.

Prediction: follow rate rises with delta at a model-specific rate; the
delta at which following recovers to the no-anchor level is the model's
verification depth. V3.2 should recover by delta=2, Sonnet later,
Llama-70B near delta=1, small Qwen models at every delta.
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


def anchored_prompt(it, j, delta):
    base = it["deep_prompt"]
    if delta is None:
        return base
    k = j - delta
    vk = it["vals"][k]
    anchor = (f" A checkpoint reading after operation {k} shows the "
              f"running value is {vk}.")
    # insert before the final instruction sentence
    marker = " Track the running value"
    return base.replace(marker, anchor + marker, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import boto3
    client = boto3.client("bedrock-runtime", region_name=args.region)
    fout = open(args.out, "a")

    for seed in args.seeds:
        rng = random.Random(5000 + seed)
        jobs = []
        for ix in range(args.n):
            it = make_item(rng, args.d, args.j)
            dd = rng.choice([x for x in range(-9, 10) if x != 0])
            vj_bad = it["vals"][args.j] + dd
            tgt = forward_from(it, args.j, vj_bad)
            for delta in args.deltas:
                k = args.j - delta
                for anch in ("anchor", "noanchor"):
                    p = anchored_prompt(it, args.j,
                                        delta if anch == "anchor" else None)
                    for cell, shown in (("edit", vj_bad),
                                        ("floor", it["vals"][args.j])):
                        lines = it["lines"][k:args.j]
                        lines[-1] = (f"After step {args.j} the value "
                                     f"is {shown}.")
                        pre = "\n".join(lines) + "\n"
                        jobs.append((ix, f"{delta}-{anch}", cell, p, pre,
                                     str(tgt), str(it["answer"])))

        def run(job):
            ix, ds, cell, p, pre, tgt, clean = job
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
                return {"exp": "r4b", "model": args.model_id, "seed": seed,
                        "item": ix, "delta": ds, "cell": cell,
                        "error": str(e)[:200]}
            ans = final_answer(text)
            return {"exp": "r4b2", "model": args.model_id, "seed": seed,
                    "item": ix, "d": args.d, "j": args.j, "delta": ds,
                    "cell": cell, "answer": ans, "clean": clean,
                    "edit_target": tgt,
                    "follows_edit": ans == tgt,
                    "follows_clean": ans == clean,
                    "text": text[:300]}

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
