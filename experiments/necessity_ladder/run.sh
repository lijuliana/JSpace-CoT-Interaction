#!/usr/bin/env bash
# necessity ladder: externalization vs correctness across the distill sizes
set -euo pipefail
cd "$(dirname "$0")/../.."
for m in 1.5B 7B 14B; do
  ml=$(echo "$m" | tr '[:upper:]' '[:lower:]')
  python src/harness/generate.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-${m}" \
    --family variable_chain --difficulties 1,2,4,8,16,32,48 \
    --n 100 --seeds 3 --out "results/raw/p1_var_${ml}.jsonl"
done
python src/analysis/curves.py results/raw/p1_var_7b.jsonl --csv results/p1_var_7b.csv --plot results/p1_var_7b.png
