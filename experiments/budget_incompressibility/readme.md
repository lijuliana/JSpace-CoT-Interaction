# Incompressibility of written values

Question: under a token budget, does the model compress by dropping values, or only by cutting prose?

Command:
```
python src/harness/api_sweep.py --model-id deepseek.v3.2 --family variable_chain \
  --difficulties 8,16,32,64 --conditions budget:64,budget:128,budget:256,budget:512,free \
  --n 30 --out results/raw/budget_var_v32.jsonl
```

Read: figures.py fig_budget.

Result (2026-08-01): prose compresses up to 2.5x with accuracy intact while externalization stays 0.98-1.00; below ~12 tokens/step the model truncates and fails rather than dropping values.
