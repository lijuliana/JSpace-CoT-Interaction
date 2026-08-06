"""J-space ablation replication on our synthetic tasks.

The anchor result this replicates: ablating the J-space (the concept
workspace read out by the Jacobian lens, Gurnee et al. 2026) hurts direct
answering more than chain-of-thought answering, the signature of written
tokens substituting for workspace state. Prior versions of this test in the
project used a blunt resample lesion and could not run as designed; the
fitted lens gives a targeted ablation. We run it on our chain/box tasks
because their scalar difficulty knob and direct-answer headroom at low d are
what the direct-vs-cot asymmetry needs.

Mechanics. The fitted lens holds per-layer J_l matrices with lens logits
unembed(J_l @ h). The residual-space direction that loads concept t at layer
l is v_t = J_l^T u_t (u_t the unembedding row). At every position of every
forward pass (prefill and decode), for each layer in a band we find the
top-k concepts by lens logit, orthonormalize their preimage directions, and
project that span out of the residual (scaled by alpha). The random control
projects out a fixed random orthonormal k-dim subspace at the same layers,
positions, and alpha; per-position removed norm is logged for both arms so
dose parity is reported, not assumed.

Design: arms {clean, jlens, random} x conditions {direct, cot} x difficulty,
families variable_chain and entity_tracking. Per generation we log
hit_token_cap and unparseable separately from wrong-answer, and for direct
cells the generated token count, so closed-channel and termination artifacts
are checkable.

Calibration mode (--calibrate) runs neutral text only: perplexity ratio,
top-1 agreement, and KL vs clean, per (k, alpha), both arms. The dose rule
is chosen and frozen there before any task cell runs.
"""

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import variable_chain, entity_tracking  # noqa: E402
from harness.generate import build_prompt, extract_answer, prompt_values  # noqa: E402
from harness.protection import NEUTRAL_TEXTS  # noqa: E402
from analysis.externalization import externalization_record  # noqa: E402


class JSpaceAblator:
    """Forward hooks projecting out top-k active J-lens directions."""

    def __init__(self, model, unembed, lens, layers, k, alpha,
                 mode="jlens", seed=0):
        self.model = model
        self.U = unembed  # [vocab, d] (lm_head weight)
        self.lens = lens
        self.layers = layers
        self.k = k
        self.alpha = alpha
        self.mode = mode  # "jlens" or "random"
        self.handles = []
        self.enabled = False
        self.removed_sq = 0.0  # sum of squared removed norms
        self.removed_n = 0
        if mode == "random":
            g = torch.Generator().manual_seed(seed)
            self.rand_basis = {}
            for li in layers:
                m = torch.randn(model.config.hidden_size, k, generator=g)
                q, _ = torch.linalg.qr(m)
                self.rand_basis[li] = q  # [d, k] orthonormal

    def __enter__(self):
        for li in self.layers:
            layer = self.model.model.layers[li]
            self.handles.append(layer.register_forward_hook(self._hook(li)))
        return self

    def _project_out(self, h, Q):
        # h: [B,S,d]; Q: [d,k] orthonormal. remove alpha * QQ^T h
        coef = torch.einsum("bsd,dk->bsk", h.float(), Q)
        rem = torch.einsum("bsk,dk->bsd", coef, Q)
        self.removed_sq += float((self.alpha * rem).norm() ** 2)
        self.removed_n += h.shape[0] * h.shape[1]
        return (h.float() - self.alpha * rem).to(h.dtype)

    def _hook(self, li):
        J = self.lens.jacobians[li]

        def hook(module, inputs, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            dev = hs.device
            if self.mode == "random":
                Q = self.rand_basis[li].to(dev)
                new = self._project_out(hs, Q)
            else:
                Jd = J.to(dev)
                Ud = self.U.to(dev).float()
                # lens logits for every position: [B,S,V]
                trans = torch.einsum("bsd,ed->bse", hs.float(), Jd)
                logits = torch.einsum("bse,ve->bsv", trans, Ud)
                top = logits.topk(self.k, dim=-1).indices  # [B,S,k]
                # preimage directions v_t = J^T u_t per position.
                # J maps layer-l residual to the final basis (transport is
                # h @ J.T), so with u_t in the final basis, v_t = u_t @ J.
                u = Ud[top]  # [B,S,k,d_final]
                v = torch.einsum("bske,ed->bskd", u, Jd)
                # orthonormalize per position via QR on [d,k]
                new = hs.float()
                B, S = hs.shape[0], hs.shape[1]
                v = v.reshape(B * S, self.k, -1).transpose(1, 2)  # [BS,d,k]
                q, _ = torch.linalg.qr(v)
                coef = torch.einsum("nd,ndk->nk",
                                    new.reshape(B * S, -1), q)
                rem = torch.einsum("nk,ndk->nd", coef, q)
                self.removed_sq += float((self.alpha * rem).norm() ** 2)
                self.removed_n += B * S
                new = (new.reshape(B * S, -1) - self.alpha * rem
                       ).reshape(B, S, -1).to(hs.dtype)
            if isinstance(output, tuple):
                return (new,) + output[1:]
            return new
        return hook

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

    def removed_rms(self):
        import math
        return math.sqrt(self.removed_sq / max(1, self.removed_n))


@torch.no_grad()
def calibrate(model, tok, lens, layers, device, ks, alphas, out):
    """Neutral-text dose calibration: perplexity ratio, top-1 agreement, KL."""
    import torch.nn.functional as F
    U = model.lm_head.weight
    texts = NEUTRAL_TEXTS
    ids_list = [tok(t, return_tensors="pt").input_ids.to(device)
                for t in texts]
    clean_logits = [model(i).logits[0, :-1].float() for i in ids_list]
    clean_lp = [F.log_softmax(c, -1) for c in clean_logits]
    rows = []
    for mode in ("jlens", "random"):
        for k in ks:
            for alpha in alphas:
                abl = JSpaceAblator(model, U, lens, layers, k, alpha,
                                    mode=mode)
                tot_kl, tot_agree, tot_lp_clean, tot_lp_abl, n = 0, 0, 0, 0, 0
                with abl:
                    abl.enabled = True
                    for i, ids in enumerate(ids_list):
                        lg = model(ids).logits[0, :-1].float()
                        lp = F.log_softmax(lg, -1)
                        tgt = ids[0, 1:]
                        tot_lp_clean += float(clean_lp[i].gather(
                            -1, tgt[:, None]).sum())
                        tot_lp_abl += float(lp.gather(
                            -1, tgt[:, None]).sum())
                        tot_agree += int((lg.argmax(-1)
                                          == clean_logits[i].argmax(-1)
                                          ).sum())
                        tot_kl += float(F.kl_div(
                            lp, clean_lp[i], log_target=True,
                            reduction="sum"))
                        n += tgt.numel()
                    abl.enabled = False
                import math
                row = {"mode": mode, "k": k, "alpha": alpha,
                       "ppl_ratio": math.exp((tot_lp_clean - tot_lp_abl)
                                             / n),
                       "top1_agree": tot_agree / n,
                       "kl_per_tok": tot_kl / n,
                       "removed_rms": abl.removed_rms()}
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def classify(pred, answer, trace_tok, cap):
    """Separate termination from error: correct / wrong / unparseable /
    hit_cap flags. hit_cap and unparseable can co-occur with wrong."""
    hit_cap = trace_tok >= cap
    unparseable = (pred.strip() == "" or
                   not re.fullmatch(r"-?\w+", pred.strip()))
    correct = pred.strip().lower() == answer.strip().lower()
    return correct, hit_cap, unparseable


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-file", default="qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt")
    ap.add_argument("--layers", default="",
                    help="comma list; default = middle third of fitted layers")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--family", default="variable_chain",
                    choices=["variable_chain", "entity_tracking", "gsm8k"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--difficulties", default="1,2,4,8")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new-cot", type=int, default=640)
    ap.add_argument("--max-new-direct", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.lens import JacobianLens

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device)
    model.eval()
    lens = JacobianLens.from_pretrained(args.lens_repo,
                                        filename=args.lens_file)
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        sl = lens.source_layers
        layers = sl[len(sl) // 3: 2 * len(sl) // 3][:6]
    print(f"lens {lens}; ablating layers {layers}", flush=True)
    U = model.lm_head.weight

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.calibrate:
        calibrate(model, tok, lens, layers, device,
                  [int(x) for x in args.ks.split(",")],
                  [float(x) for x in args.alphas.split(",")], args.out)
        return

    gen = (variable_chain if args.family == "variable_chain"
           else lambda d, s: entity_tracking(5, d, s))
    gsm = None
    if args.family == "gsm8k":
        # assignment-dataset mode: difficulty axis is a single pooled set,
        # gold answers from the dataset, chat template with /no_think so
        # cot is a worked solution and direct is answer-only
        gsm = [json.loads(l) for l in
               open(os.path.join(args.data_dir, "gsm8k_test.jsonl"))]
        diffs = [0]
    else:
        diffs = [int(x) for x in args.difficulties.split(",")]
    arms = ["clean", "jlens", "random"]
    out = open(args.out, "w")

    def make_prompt_answer(cond, d, s):
        if gsm is not None:
            row = gsm[s]
            gold = row["answer"].split("####")[-1].strip().replace(",", "")
            q = row["question"]
            suffix = ("\nGive only the final answer, nothing else. Do not "
                      "show any working. /no_think" if cond == "direct"
                      else "\nSolve step by step, then end with "
                      "'#### <number>'. /no_think")
            p = tok.apply_chat_template(
                [{"role": "user", "content": q + suffix}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            return p, gold, None
        inst = gen(d, s)
        return build_prompt(inst, cond, tok, False), inst.answer, inst

    for arm in arms:
        for cond in ["direct", "cot"]:
            cap = (args.max_new_direct if cond == "direct"
                   else args.max_new_cot)
            for d in diffs:
                for s in range(args.n):
                    prompt, answer, inst = make_prompt_answer(cond, d, s)
                    ids = tok(prompt, return_tensors="pt").to(device)
                    if arm == "clean":
                        o = model.generate(
                            **ids, max_new_tokens=cap, do_sample=True,
                            temperature=0.6, top_p=0.95,
                            pad_token_id=tok.eos_token_id)
                        rms = 0.0
                    else:
                        abl = JSpaceAblator(model, U, lens, layers,
                                            args.k, args.alpha, mode=arm)
                        with abl:
                            abl.enabled = True
                            o = model.generate(
                                **ids, max_new_tokens=cap, do_sample=True,
                                temperature=0.6, top_p=0.95,
                                pad_token_id=tok.eos_token_id)
                            abl.enabled = False
                        rms = abl.removed_rms()
                    gen_tok = o.shape[1] - ids.input_ids.shape[1]
                    text = tok.decode(o[0][ids.input_ids.shape[1]:],
                                      skip_special_tokens=True)
                    if gsm is not None:
                        m = re.findall(r"####\s*\$?(-?[\d,]+)", text)
                        pred = (m[-1].replace(",", "") if m
                                else extract_answer(text, cond))
                    else:
                        pred = extract_answer(text, cond)
                    correct, hit_cap, unparseable = classify(
                        pred, answer, gen_tok, cap)
                    ext = None
                    if cond == "cot" and args.family == "variable_chain":
                        # externalization under ablation: does a targeted
                        # internal squeeze change how much gets written
                        e = externalization_record(
                            text, inst.intermediates, prompt_values(inst))
                        ext = e["externalization_fraction"]
                    out.write(json.dumps({
                        "arm": arm, "condition": cond, "difficulty": d,
                        "seed": s, "k": args.k, "alpha": args.alpha,
                        "layers": layers, "correct": correct,
                        "hit_cap": hit_cap, "unparseable": unparseable,
                        "gen_tokens": gen_tok, "removed_rms": rms,
                        "ext_frac": ext, "pred": pred[:40],
                    }) + "\n")
                    out.flush()
                print(f"{arm}/{cond}/d={d} done", flush=True)
    out.close()


if __name__ == "__main__":
    main()
