# Proposed captions

Takeaway first, then encoding. Error bars are 95% intervals (Wilson for
rates, bootstrap over items for model comparisons) unless stated otherwise.

The shared protocol, defined once here and named in each caption: the model
solves a chained arithmetic problem while writing its steps ("After step 5
the value is 457."). We change one written intermediate value (an **edit**
changes the visible text; an **injection** overwrites the hidden state at
that token while the text stays unchanged), let the model continue, and
measure the fraction of runs whose final answer builds on the changed value
rather than the original.

**F1 (capacity).** Writing intermediate values is not a response to running
out of internal capacity. (A) When the model must answer without writing
any reasoning ("silent"), accuracy collapses after one to two dependent
steps (solid: DeepSeek-R1-671B; dashed: R1-distill-7B); with writing
allowed it stays near ceiling (n=30 items per depth). (B) The fraction of
intermediate values the model writes into its reasoning is 1.00 at every
depth, including depth 1, so writing does not switch on at some difficulty
threshold (30 to 1,920 values per depth; 95% Wilson intervals). (C) No
intervention moves state between the internal activations and the written
text: deleting the J-space reading of the current value, and matched random
directions, leave accuracy unchanged (GSM8K, Qwen3-4B); damaging the
residual stream during generation hurts a task whose state stays internal
(box tracking) three times more than the written chain task; and capping
the written text below roughly twelve tokens per step collapses accuracy by
truncation rather than pushing the values into internal memory (DeepSeek
V3.2, d=16).

**F2 (registers).** The hidden state under a written value behaves as a
settable register, and two registers compose. (A) The injection: the text
still says 457, but the hidden state at that token now encodes 462, a
number that appears nowhere; the final answer continues from 462. (B) The
fraction of runs whose final answer uses the injected value, on six models;
restoring the correct value recovers the correct answer, and a norm-matched
random direction never produces the target. (C, D) Two counters A and B run
in one problem; injecting a replacement for either or both (a′, b′) makes
the final answer equal the matching sum (a′+B, A+b′, or a′+b′), while
answers using only one of two injected values occur at 0.01 (30 items, five
samples each).

**F3 (retrieval).** Later computation retrieves a written value by
attending to its token, through the middle third of the network's layers.
(A) Preventing all later positions from attending to the edited value's
token drops edit use to zero on four models, while blocking a neighboring
token, a random prompt token, or a random token elsewhere in the reasoning
changes nothing (n=600 per condition per model). (B) Injecting at any
single layer up to 22 works; from layer 24 on it does nothing (Qwen3-4B, 36
layers, n=60). (C) Blocking attention to the value token across layers 12
to 23 (shaded in both panels) removes 0.83 of edit use; blocking any single
layer removes none (gray dots, n=80), so the read is spread across the
middle third and finished by layer 23.

**A (layer cliff, standalone).** Overwriting the hidden state works at
every layer before the attention read completes and does nothing after:
the fraction of runs using the injected value falls from 0.93 at layer 22
to 0.01 at layer 24 (Qwen3-4B, 36 layers, n=60). The shaded band, layers 12
to 23, is the only region where blocking attention to the value token
collapses edit use.

**B (composition, standalone).** Two written counters are independently
settable and compose. (A) Replacement values a′ and b′ are injected into
the hidden state only; the text still says 512 and 347, yet the final
answer is a′+b′ = 641, a number written nowhere. (B) The expected sum
appears at 0.88 to 0.95 on both models for every injection pattern; a
norm-matched random injection moves the answer off every expected value,
and answers using only one of two injected values occur at 0.01.

**C (per-model injection, standalone).** The hidden state under a written
value is a settable register on all six models tested: injecting a
never-written value makes the final answer build on it at 0.70 to 1.00,
restoring the correct value recovers the correct answer, and a norm-matched
random direction never produces the target value.

**D (protocol).** The edit-and-continue protocol. One written step of the
model's own reasoning is altered and the model continues from it; the
readout is whether the final answer builds on the altered value. The three
interventions: (1) edit the visible text of the step; (2) overwrite the
hidden state at the value's token while the text stays unchanged; (3) leave
everything intact but block later positions from attending to the value's
token.

**E (reverting heatmap).** Models undo an edited value only when the prompt
gives them the material to recompute it, and which evidence triggers
recomputation differs by model. Cells: fraction of runs in which the model
abandons the edited value and returns to the correct one, by what the
prompt provides about that value; dashes mark conditions not run for that
model.

**F4 (read vs. recompute).** Whether a model trusts an edited written value
tracks what it can recompute, not model size. Blue: the final answer builds
on the edit. Vermilion: the model recomputes the correct value from the
prompt and ignores the edit. DeepSeek V3.2 recomputes exactly the values
whose derivation the prompt states (0.57 vs. 0.93 to 0.96); Claude Sonnet
4.5 recomputes a step when it is the value's only source (0.22 vs. 0.81);
Llama 3.3 70B recomputes neither despite solving the same problems from
scratch; on GSM8K, whose intermediate values any strong model can rederive
in one step, Qwen3-4B still follows corrupted values at 0.87 while DeepSeek
V3.2 follows at 0.10.
