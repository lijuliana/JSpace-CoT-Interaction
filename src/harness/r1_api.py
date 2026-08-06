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

LLAMA_TMPL = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
              "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
              "\n\n{prefill}")


def llama_invoke(client, model_id, user_text, prefill, max_tokens):
    """Bedrock Llama has no Converse prefill; continue via raw template."""
    body = json.dumps({
        "prompt": LLAMA_TMPL.format(user=user_text, prefill=prefill),
        "max_gen_len": max_tokens, "temperature": 0.6, "top_p": 0.95})
    for attempt in range(6):
        try:
            r = client.invoke_model(modelId=model_id, body=body)
            return json.loads(r["body"].read())["generation"]
        except Exception:
            if attempt == 5:
                raise
            import time
            time.sleep(2 ** attempt)
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
            # offtarget control: generative sentence present (shallow
            # prompt defines v_j), but the edit is at j2 = j + 2, a value
            # the prompt does not define. Targeted verification predicts
            # following here matches deep; global re-solving predicts
            # reversion like shallow.
            j2 = it["j"] + 2
            v2_bad = it["vals"][j2] + delta
            lines = it["lines"][:j2]
            lines[-1] = f"After step {j2} the value is {v2_bad}."
            pre2 = "\n".join(lines) + "\n"
            jobs.append((ix, "offtarget", "edit", it["shallow_prompt"],
                         pre2, forward_from(it, j2, v2_bad)))
            lines[-1] = f"After step {j2} the value is {it['vals'][j2]}."
            jobs.append((ix, "offtarget", "floor", it["shallow_prompt"],
                         "\n".join(lines) + "\n",
                         forward_from(it, j2, v2_bad)))

        def run(job):
            ix, rend, cell, p, pre, tgt = job
            it = items[ix]
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
                return {"exp": "r1api", "model": args.model_id,
                        "seed": seed, "item": ix, "rendering": rend,
                        "cell": cell, "error": str(e)[:200]}
            ans = final_answer(text)
            return {"exp": "r1api", "model": args.model_id, "seed": seed,
                    "item": ix, "d": args.d, "j": it["j"],
                    "rendering": rend, "cell": cell, "answer": ans,
                    "clean": str(it["answer"]), "edit_target": str(tgt),
                    "follows_edit": ans == str(tgt),
                    "follows_clean": ans == str(it["answer"]),
                    "text": text[:300]}

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
