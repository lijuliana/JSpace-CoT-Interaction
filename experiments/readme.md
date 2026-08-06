# Experiments

Each directory is one experiment: what it asks, the command that produced its data, and how to read the result. Raw data lives in results/raw (gitignored, regenerable from these commands); figures in results/figures.

Order roughly as the argument builds:

1. `necessity_ladder` - is complete externalization necessary for correct deep chains? (1.5B/7B/14B)
2. `gate_b_readback` - are written values read back, and does the internal copy verify them?
3. `budget_incompressibility` - under token pressure, does the model drop values or just prose?
4. `gate_a_capacity` - does squeezing internal capacity change how much gets written?
5. `eviction` - does a value's internal decodability fall after it is written?
6. `protection` - does CoT shield accuracy from internal lesions, more for serial than parallel tasks?
7. `capacity_law` - how deep a serial computation fits in one forward pass, vs model depth?

See notes/synthesis.md for how they combine into the claim, and notes/results-log.md for dated findings.
