# Necessity of externalization

Question: among traces that get the right answer on a serial chain, do any omit intermediate values? If complete externalization is necessary for depth, the answer is no.

Command:
```
python src/harness/generate.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-{1.5B,7B,14B} \
  --family variable_chain --difficulties 1,2,4,8,16,32,48 --n 100 --seeds 3 --out results/raw/p1_var_<m>.jsonl
```

Read: `python src/analysis/curves.py results/raw/p1_var_7b.jsonl --csv ...`, then split externalization fraction by correctness (see figures.py fig_necessity).

Result (2026-08-02): externalization fraction among correct traces at d>=16 is exactly 1.000 across all three sizes (n=1935 pooled); among wrong traces ~0.5. No correct deep chain omits a value.
