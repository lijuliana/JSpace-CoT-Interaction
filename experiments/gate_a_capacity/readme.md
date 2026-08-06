# Gate A: does internal pressure change writing (dropped)

Intended: lesion a mid-stack window during generation and check whether externalization rises (compensation). Dropped: the coarse dose sweep overshot the accuracy cliff, the write policy proved saturated so there was little compensation headroom to detect, and the question was superseded by the read-back patch (which gives the clean causal result). The lesion machinery (src/harness/gate_a_lesion.py) is reused by the protection experiment.
