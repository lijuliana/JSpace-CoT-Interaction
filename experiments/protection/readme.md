# Protection: the serial/parallel dissociation, causally

Question: does squeezing the internal residual stream hurt entity tracking (state held internally) more than variable chains (state written on the page)?

Command (final, corrected):
```
python src/harness/protection.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --family {entity_tracking,variable_chain} --difficulties 2,4,8,16 --n 40 --alpha 0.10 \
  --max-new 1000 --out results/raw/protection_<fam>_7b_fix10.jsonl
```
Arms: clean / target layer window / matched-damage control window. The lesion fires during prefill and decode (an earlier version fired only on decode, which barely touched the direct condition and made cot-vs-direct uninterpretable; fixed). Damage is metered on neutral text and against the control arm, since neither KL number is clean.

Read: `python src/analysis/protection_readout.py --entity <ent> --chain <chain>`

Result (2026-08-03): where both families have accuracy headroom (d=2,4), the internal lesion drops entity-tracking cot accuracy ~3x more than variable-chain cot accuracy (0.34 vs 0.11), robust to the lesioned window. Honest caveat: the lesion is blunt (it also damages re-reading of written values), so at d=8 it hurts chains too while entity has floored; the clean comparison sits at low-to-mid difficulty. Supporting evidence for the dissociation; the read-back patch is the clean causal claim.
