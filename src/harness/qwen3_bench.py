"""Assignment-constraint core: Qwen3-4B on GSM8K, MATH-500, and AIME 2024.

Three modes, all vLLM-batched.

sweep: free (thinking) vs direct (no-think, answer only) accuracy on all
three datasets, the intended difficulty ladder. Logs cap hits, unparseable
answers, and generated token counts (direct cells prove channel closure).

readback: the corruption test on GSM8K worked solutions. Elicit a no-think
worked solution with each computed value on its own line, corrupt one
mid-trace computed value that is not a given, continue from the corrupted
prefix, and compare against a no-corruption resample control from the same
clean prefix.

Datasets: GSM8K test (openai/grade-school-math), MATH-500
(HuggingFaceH4/MATH-500 rev 6e4ed1a2a79a), AIME 2024 I and II, 30 problems
(HuggingFaceH4/aime_2024 rev 2fe88a2f1091). Model Qwen/Qwen3-4B rev
1cfa9a720891.
"""

import argparse
import json
import os
import re

DIRECT = ("\nGive only the final answer, nothing else. Do not show any "
          "working. /no_think")
SOLVE_GSM = ("\nSolve step by step. Show each calculation on its own line "
             "ending with '= <value>'. End with '#### <number>'. /no_think")


def load_ds(data_dir):
    ds = {}
    ds["gsm8k"] = [json.loads(l) for l in
                   open(os.path.join(data_dir, "gsm8k_test.jsonl"))]
    ds["math500"] = [json.loads(l) for l in
                     open(os.path.join(data_dir, "math500.jsonl"))]
    ds["aime2024"] = [json.loads(l) for l in
                      open(os.path.join(data_dir, "aime2024.jsonl"))]
    return ds


def gold(ds_name, row):
    if ds_name == "gsm8k":
        return row["answer"].split("####")[-1].strip().replace(",", "")
    return str(row["answer"]).strip()


def question(ds_name, row):
    return row["question"] if ds_name == "gsm8k" else row["problem"]


def norm(s):
    s = s.strip().strip("$").replace(",", "").replace(" ", "")
    s = re.sub(r"\\!|\\,|\\;", "", s)
    m = re.fullmatch(r"\\d?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    if m:
        try:
            return str(float(m.group(1)) / float(m.group(2)))
        except ZeroDivisionError:
            return s
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s.lower()


def extract(text):
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1]
    m = re.findall(r"####\s*\$?(-?[\d,./]+)", text)
    if m:
        return m[-1]
    m = re.findall(r"[Aa]nswer\s*(?:is)?\s*[:=]?\s*\$?(-?[\d,./]+)", text)
    if m:
        return m[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text[-200:])
    return nums[-1] if nums else ""


def grade(ds_name, pred, g):
    return norm(pred) == norm(g)


def chat(tok, text, thinking):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "readback"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-gsm", type=int, default=150)
    ap.add_argument("--n-math", type=int, default=150)
    ap.add_argument("--n-readback", type=int, default=80)
    ap.add_argument("--delta", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, gpu_memory_utilization=0.9,
              max_model_len=20000)
    ds = load_ds(args.data_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = open(args.out, "w")

    if args.mode == "sweep":
        caps = {"gsm8k": 2048, "math500": 6144, "aime2024": 16000}
        ns = {"gsm8k": args.n_gsm, "math500": args.n_math, "aime2024": 30}
        for name in ["gsm8k", "math500", "aime2024"]:
            rows = ds[name][:ns[name]]
            for cond in ["direct", "free"]:
                if cond == "direct":
                    prompts = [chat(tok, question(name, r) + DIRECT, False)
                               for r in rows]
                    sp = SamplingParams(temperature=0.0, max_tokens=32)
                else:
                    prompts = [chat(tok, question(name, r), True)
                               for r in rows]
                    sp = SamplingParams(temperature=0.6, top_p=0.95,
                                        max_tokens=caps[name])
                outs = llm.generate(prompts, sp)
                for r, o in zip(rows, outs):
                    text = o.outputs[0].text
                    g = gold(name, r)
                    pred = extract(text)
                    gen_tok = len(o.outputs[0].token_ids)
                    cap = caps[name] if cond == "free" else 32
                    out.write(json.dumps({
                        "dataset": name, "condition": cond,
                        "gold": g, "pred": pred[:60],
                        "correct": grade(name, pred, g),
                        "gen_tokens": gen_tok,
                        "hit_cap": gen_tok >= cap,
                        "unparseable": pred == "",
                    }) + "\n")
                out.flush()
                print(f"{name}/{cond} done", flush=True)

    else:  # readback on gsm8k worked solutions
        rows = ds["gsm8k"][:args.n_readback * 2]
        prompts = [chat(tok, question("gsm8k", r) + SOLVE_GSM, False)
                   for r in rows]
        sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=900)
        outs = llm.generate(prompts, sp)
        cont_prompts, resample_prompts, metas = [], [], []
        for r, o, p in zip(rows, outs, prompts):
            text = o.outputs[0].text
            g = gold("gsm8k", r)
            if not grade("gsm8k", extract(text), g):
                continue
            qnums = set(re.findall(r"\d+", question("gsm8k", r)))
            body = re.split(r"####|\\boxed", text)[0]
            lines = body.splitlines()
            cand = [(i, m.group(1)) for i, ln in enumerate(lines)
                    if (m := re.search(r"=\s*\$?(-?\d+)\s*$", ln.strip()))
                    and m.group(1) not in qnums and i < len(lines) - 1]
            if not cand:
                continue
            idx, val = cand[len(cand) // 2]
            corr = str(int(val) + args.delta)
            cut = lines[:idx] + [re.sub(r"(=\s*\$?)-?\d+(\s*)$",
                                        rf"\g<1>{corr}\g<2>", lines[idx])]
            cont_prompts.append(p + "\n".join(cut))
            resample_prompts.append(p + "\n".join(lines[:idx + 1]))
            metas.append((g, val, corr, extract(text)))
        sp2 = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=700)
        conts = llm.generate(cont_prompts, sp2)
        resams = llm.generate(resample_prompts, sp2)
        for (g, val, corr, clean_pred), c, rs in zip(metas, conts, resams):
            corr_pred = extract(c.outputs[0].text)
            res_pred = extract(rs.outputs[0].text)
            out.write(json.dumps({
                "gold": g, "orig_val": val, "corr_val": corr,
                "clean_pred": clean_pred, "corr_pred": corr_pred[:40],
                "resample_pred": res_pred[:40],
                "answer_changed": norm(corr_pred) != norm(clean_pred)
                and corr_pred != "",
                "resample_changed": norm(res_pred) != norm(clean_pred)
                and res_pred != "",
                "unparseable": corr_pred == "",
            }) + "\n")
        out.flush()
        print(f"readback: {len(metas)} corrupted items", flush=True)
    out.close()


if __name__ == "__main__":
    main()
