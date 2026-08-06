# Read-back and verification

Question: corrupt the last written mention of a mid-chain value, continue generation. Does the answer follow the corrupted value (read-back) or the clean value (internal copy)?

Command:
```
python src/harness/gate_b_corruption.py --model <M> --family variable_chain \
  --difficulties 4,8,16,32 --n 150 --out results/raw/gate_b_var_<m>.jsonl
```

Read: `python src/analysis/gate_b_readout.py results/raw/gate_b_var_7b.jsonl`

Result (2026-08-01): follows_corruption 0.31->0.50 rising to d=16 then 0.36 at d=32, where restates_clean jumps to 0.45 (model notices the edit). Written values are causally live. The follows-clean share is re-derivation from the in-context prompt, not a persistent internal copy (see results-log 2026-08-03); the d=32 dip reflects the post-think re-solve and selection toward easier surviving instances.
