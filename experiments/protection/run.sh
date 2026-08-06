#!/usr/bin/env bash
# protection: matched-dose lesion on serial vs parallel families (corrected lesion:
# fires during prefill and decode)
set -euo pipefail
cd "$(dirname "$0")/../.."
for fam in entity_tracking variable_chain; do
  python src/harness/protection.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --family "$fam" --difficulties 2,4,8,16 --n 40 --alpha 0.10 \
    --max-new 1000 --out "results/raw/protection_${fam}_7b_fix10.jsonl"
done
python src/analysis/protection_readout.py \
  --entity results/raw/protection_entity_tracking_7b_fix10.jsonl \
  --chain results/raw/protection_variable_chain_7b_fix10.jsonl
