# Proposed captions

Takeaway first, then encoding. Error bars are 95% intervals (Wilson for
rates, bootstrap over items for model comparisons) unless stated otherwise.

**F1 (capacity).** Writing intermediate values is not a response to running
out of internal capacity. (a) With the written chain of thought suppressed,
accuracy collapses after one to two dependent steps (solid: DeepSeek-R1-671B;
dashed: R1-distill-7B); with writing allowed it stays near ceiling (n=30
items per depth). (b) The fraction of intermediate values that appear in the
trace is 1.00 at every depth, including depth 1 (30 to 1,920 values per
depth; error bars are 95% Wilson intervals). (c) No intervention moves state
between the internal workspace and the trace: J-space ablation and matched
random directions leave accuracy unchanged (GSM8K, Qwen3-4B), residual
lesions hurt the internally held box-tracking task three times more than the
written chain task, and a token cap below the per-step floor collapses
accuracy by truncation rather than inducing internal holding (DeepSeek V3.2,
d=16).

**F2 (registers).** Written positions behave as settable, compositional
registers. (a) The hidden state at a written value token is overwritten with
the state of a value never written anywhere; the visible text is unchanged.
(b) The answer follows the planted value on six models; restoring the
correct state and a norm-matched random control bound the effect. Bars are
95% intervals over items. (c, d) With two counters A and B in one text,
planting a′, b′, or both yields the matching sum (a′+B, A+b′, or a′+b′) as
the dominant answer category; answers combining only one planted value occur
at 0.01 (30 items, five samples each).

**F3 (retrieval).** Written values are retrieved through middle-layer
attention to the value token. (a) Blocking attention from post-edit
positions to the edited value token collapses edit-following on four models;
blocking a neighboring token, a random prompt token, or a random trace token
does not (n=600 per condition per model). (b) Overwriting the residual
stream at one single layer suffices at every layer up to 22 and is inert
from layer 24 on (Qwen3-4B, 36 layers, n=60). (c) Blocking attention to the
value token across the middle third of layers (12 to 23) removes 0.83 of
edit-following; no single layer is necessary (gray dots, n=80). The shaded
band in (b) and (c) marks layers 12 to 23.

**A (layer cliff, standalone).** A single-layer state patch works at every
layer before the attention read completes and is inert after it: the rate at
which the answer follows the planted value falls from 0.93 at layer 22 to
0.01 at layer 24 (Qwen3-4B, 36 layers, n=60 items). The shaded band marks
layers 12 to 23, the only region where attention knockout collapses
edit-following.

**B (composition, standalone).** Two written counters are independently
settable and compose. (a) Values a′ and b′ are planted into the hidden state
only; the visible text still says 512 and 347. (b) The answer equals the sum
of whatever combination was planted at 0.88 to 0.95 on both models; a
norm-matched random patch moves the answer off every expected value, and
answers using only one planted value occur at 0.01.

**C (per-model patch, standalone).** The residual-stream state at a written
value token is a settable register on all six models tested: planting a
never-written value moves the answer to it at 0.70 to 1.00, restoring the
correct state recovers the correct answer, and a norm-matched random patch
does not produce the target value. Bars are 95% intervals over items.

**D (protocol).** The edit-and-continue protocol. A written intermediate
value is corrupted, the continuation is regenerated, and the readout is
whether the final answer follows the planted value. The three interventions:
(1) edit the visible text, (2) overwrite the hidden state at the value token
while leaving the text unchanged, (3) block attention from later positions
to the value token.

**E (reverting heatmap).** Models revert an edited value to the correct one
only when the prompt supplies enough evidence to recompute it. Cells give
the rate of reverting the edit; gray dashes mark conditions not run for that
model.

**F4 (read vs. recompute).** Whether a model trusts an edited written value
tracks what it can recompute, not model size. Blue: the answer follows the
edit. Vermilion: the model reverts to the correct value. DeepSeek V3.2
recomputes exactly the values the prompt defines (0.57 vs. 0.93 to 0.96);
Claude Sonnet 4.5 recomputes checkpoints that are the sole source of a value
(0.22 vs. 0.81); Llama 3.3 70B checks neither; on GSM8K, Qwen3-4B follows
corrupted intermediates at 0.87 while V3.2, which can recompute them,
follows at 0.10.
