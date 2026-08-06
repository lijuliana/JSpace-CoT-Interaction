# Eviction vs redundant cache

Question: with writing made exogenous (teacher-forced traces that either write or omit one value), is the value less decodable from the current hidden state after it was written than after it was suppressed?

Command:
```
python src/harness/eviction_probe.py --model <M> --depth 12 --n 400 --out results/raw/eviction_<m>.jsonl
```

Read: `python src/analysis/eviction_readout.py results/raw/eviction_7b_d12.jsonl`. Negative written-minus-suppressed gap = eviction; non-negative = redundant cache.
