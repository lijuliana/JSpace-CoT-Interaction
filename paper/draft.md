# Written Values Are Read Back: The Chain of Thought as Addressable Memory

## Abstract

When a language model writes an intermediate value during multi-step
reasoning, does the written token carry the computation, or is it a
commentary on work done elsewhere? We answer this causally on arithmetic
chain tasks where every intermediate value has known ground truth.
Planting a value the model never wrote into the residual-stream state at a
written token makes the final answer track that value in 74 to 98 percent
of cases across three unrelated model families (Qwen, Phi, OLMo) and
chain lengths up to forty steps, while a norm-matched random perturbation
moves answers in under 1 percent. Blocking attention from later positions
to the written value's token drops edit-following from 0.90 to 0.00, while
blocking a neighboring token, a random prompt token, or an earlier written
value changes nothing. Two values planted in two positions compose: the
model reports their sum, which appears nowhere in its input, in 88 percent
of cases. The written token's position therefore behaves as memory that
later computation reads through attention. Models consult this memory by
default: edited values are followed even when the correct value is one
operation away in the prompt, and which prompt evidence can override the
written record differs by model. DeepSeek V3.2 rederives a
value only when the prompt itself defines it (reversion 0.43 versus 0.05
for a stated checkpoint it would only need to copy); Claude Sonnet 4.5
also uses stated checkpoints when they are the only source (reversion
0.78); Llama 3.3 70B rarely checks anything despite having the capacity.
For oversight, the consequence is direct: on state a model cannot cheaply
regenerate, the visible trace is causally load-bearing, and this holds
more, not less, for stronger models.

## 1. Introduction

Chain-of-thought monitoring assumes the text a model writes reflects the
computation it performs. Faithfulness studies have shown the two can
disagree, but disagreement in specific cases leaves the general mechanism
open: when a model writes "After step 5 the value is 452" and later uses
452, did the written token cause the later use?

We give a mechanistic answer for arithmetic chain tasks. The finding is
that the written value is stored, at its own token position, in a form
that later computation reads back through attention, and that this stored
value, not the visible text and not the correct value, determines the
final answer.

Three interventions on the same protocol establish this.

First, planting values. We overwrite the residual-stream activations at
one written value's token with the activations that value's line would
have had if a different number had been written, leaving the visible text
unchanged. The final answer follows the planted number, which appears
nowhere in the text, at 0.74 (Qwen2.5-7B), 0.94 (Phi-3-medium), 0.93
(OLMo-2-7B) at ten steps, and 0.95 to 0.98 for Qwen3-4B at five to forty
steps. A random vector of the same size moves answers at 0.004 or less.
Restoring the correct state while the text shows a corrupted value returns
the answer to correct at 0.86 to 1.00. Planting two different unwritten
values at two positions yields their sum at 0.88 to 0.90, at or above the
product of the single-position rates, so positions act as independent
slots.

Second, cutting the read. Blocking attention from all positions after an
edited value to that value's token collapses edit-following from 0.90 to
0.000 (Qwen3-4B; 0.64 to 0.000 on Qwen2.5-7B). Blocking the adjacent
tokens on the same line, a random operand in the prompt, or an earlier
written value leaves following at baseline (0.64 to 0.90). Making the
value's characters unreadable without touching attention sends the model
back to rederiving from the previous line (60 to 73 percent recover the
correct answer), so attention read-back is the preferred path, not the
only possible one.

Third, testing when the written record is checked. Models follow edited
values by default. Whether any prompt evidence overrides the written
record depends on the kind of evidence and on the model. Across four
conditions (the correct value is implicitly derivable from the prompt; a
checkpoint sentence states it redundantly; a checkpoint sentence is its
only source; the prompt generatively defines it), DeepSeek V3.2 reverts
to the correct answer only in the last condition (0.43 versus at most
0.05 elsewhere), Claude Sonnet 4.5 reverts most strongly when the
checkpoint is the sole source (0.78) and diffusely elsewhere (0.25 to
0.32), and Llama 3.3 70B follows edits at 0.97 in nearly every condition.
Qwen models at 4B and 7B never revert (gap under 0.01 between conditions).
Capacity to rederive is necessary for checking; whether it is used is a
property of the model's training.

The oversight consequence follows from the mechanism. On values a model
cannot cheaply regenerate from the prompt, the trace is not commentary;
it is the memory the model computes with, and edit-following on such
values increases with model strength (R1-distill-7B 0.42, DeepSeek V3.2
0.78, Claude Sonnet 4.5 0.97 on the same protocol through API prefill).

## 2. Tasks and protocol

All causal experiments use arithmetic chain problems: start with a
three-digit number, apply d signed two-digit additions and subtractions,
report the final value. Every intermediate value is determined by
construction, so the effect of any edit is computable exactly. The worked
solution writes one line per step ("After step i the value is v"). The
edit-and-continue protocol takes a correct worked solution, changes the
value at step j by a small amount, truncates immediately after the edited
line, and lets the model continue; a no-edit truncation measures the
sampling noise floor, which is 0.00 throughout. The final answer is
parsed from a fixed format, and unparseable answers are logged separately.

Three interventions build on this base. The text edit changes only the
visible token. The state patch leaves the text intact and overwrites the
residual stream at the value's token position across a middle band of
layers with activations captured from a run in which a different value
was written there. The attention block adds a mask so that no position
after the edit can attend to the value's token, at all layers.

White-box experiments run on Qwen3-4B, Qwen2.5-7B-Instruct,
Phi-3-medium, OLMo-2-7B-Instruct, and DeepSeek-R1-Distill-Qwen 7B and
14B. Frontier
models (DeepSeek V3.2, Claude Sonnet 4.5, Llama 3.3 70B) are tested
behaviorally through API prefill. Sample sizes are 100 to 600 per
condition with 2 to 3 seeds; error bars are 95 percent bootstrap
intervals. Model revisions, prompts, and configs are in the appendices.

## 3. The state at a written token is addressable memory

| model | text edit followed | restore correct state | plant never-written value | random control | n |
|---|---|---|---|---|---|
| Qwen3-4B | 0.98 [0.95, 1.00] | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 106 |
| Qwen2.5-7B-Instruct | 0.85 [0.80, 0.90] | 0.97 [0.92, 1.00] | 0.74 [0.67, 0.81] | 0.00 [0.00, 0.01] | 93 |
| Phi-3-medium | 0.94 [0.90, 0.98] | 0.98 [0.95, 1.00] | 0.94 [0.91, 0.97] | 0.00 [0.00, 0.00] | 102 |
| OLMo-2-7B | 0.85 [0.79, 0.91] | 0.88 [0.82, 0.94] | 0.93 [0.88, 0.97] | 0.00 [0.00, 0.00] | 100 |
| R1-distill-7B | 0.60 [0.53, 0.66] | 1.00 [0.99, 1.00] | 0.76 [0.68, 0.84] | 0.30 [0.22, 0.39] | 65 |
| R1-distill-14B | 0.44 [0.36, 0.52] | 0.99 [0.98, 1.00] | 0.70 [0.61, 0.80] | 0.21 [0.13, 0.29] | 46 |

Brackets are 95 percent bootstrap intervals; rates conditioned on items
where the text edit was followed; ten-step chains. The distill models
re-solve the problem after their reasoning phase, which elevates their
random-control reversion; re-solving can only produce the correct answer,
so their planted-value rates are unaffected by it. R1-distill-1.5B is
excluded (12 usable items; random control exceeds the planted-value
rate).

Editing the visible text alone changes the final answer on most items
(0.85 to 0.94 across the three instruct families at ten steps). The state
patch decomposes this effect. Restoring the correct internal state under
corrupted text returns the answer to correct (0.97, 0.98, 0.88 for
Qwen2.5-7B, Phi-3-medium, OLMo-2-7B). Planting a third value that appears
nowhere in text makes the answer track it (0.74, 0.94, 0.93). The
norm-matched random control is at 0.004 or below in every cell, so the
effect is carried by the value content of the state, not by perturbation
size. The effect persists across depth: Qwen3-4B follows the planted
value at 0.984, 0.974, 0.951 for chains of five, twenty, and forty steps.

The mechanism is not specific to the arithmetic template. On a
natural-language state task, in which a story character's count of
marbles or stamps changes through varied narrative events and the trace
lines vary in phrasing, the full pattern replicates: text edits are
followed at 0.70 and 0.74 (Qwen3-4B, Phi-3-medium), restoring the
correct state returns the answer to correct at 0.89 and 0.94, a planted
never-written value is followed at 0.83 [0.75, 0.90] and 0.86 [0.79,
0.92], and the norm-matched random control is 0.00 in both (n=86 and 89
conditioned items).

Two slots compose. In a task with two independent counters summed at the
end, planting unwritten values at both final positions produces their sum
(0.90 and 0.88 on Qwen2.5-7B and Qwen3-4B), matching the product of the
single-slot rates, while the same-norm random patch leaves the clean
answer in place (0.95). Mixed sums, in which one planted value is read
and the other ignored, occur at 0.01, so the two positions are read
independently. The model reports a number that exists nowhere in its
input, assembled from two independently planted memories.

## 4. Read-back runs through attention to the value's token

[Figure: knockout bars]

Blocking attention from post-edit positions to the edited value's tokens
collapses following on every model tested, across three families:
0.643 to 0.000 (Qwen2.5-7B), 0.900 to 0.000 (Qwen3-4B), 0.880 to 0.000
(Phi-3-medium), 0.333 to 0.005 (OLMo-2-7B, whose baseline is low in this
prompt format but whose contrast is intact); n=600 per condition. The
same collapse holds on the natural-language task (0.730 to 0.007 on
Qwen3-4B, n=300). The same block applied to the adjacent
words on the same line, to a random operand in the prompt, or to an
earlier step's written value leaves following at baseline. Blocking the
operand that the next step needs collapses arithmetic itself (following
0.03, with answers matching neither the edited nor the correct value).
Replacing the value's characters with dots, without touching attention,
also removes following, but differently: Qwen models reconstruct the
value from the previous line and return the correct answer 60 to 73
percent of the time, while Phi-3-medium does not recover (0.00), so the
fallback path exists in some models and not others. Attention to the written token is how the value normally moves
forward; when the token is unreadable, the model can rebuild it, and by
default it does not.

## 5. What evidence overrides the written record

[Figure: four-condition reversion, V3.2 and Sonnet]

By default the written record wins. In the deep condition, where
rederiving the edited value from the prompt requires replaying the chain,
every model follows the edit (0.89 to 0.97 at ten steps). Making the
correct value one operation away changes little by itself: V3.2 follows
edits at 0.85 even when the edited value is one step from the prompt's
starting number, and Sonnet follows at 0.98 in the same position.

What matters is the kind of evidence, and models differ sharply. The
conditions below come from separate experiments with different trace
lengths and item sets; each within-experiment contrast is controlled, and
we do not claim a single ordered scale across them.

DeepSeek V3.2 reverts only when the prompt generatively defines the
value (a sentence gives the two numbers whose sum becomes the running
value): reversion 0.43, against 0.01 when a checkpoint sentence states
the same value redundantly and 0.05 when the checkpoint is the value's
only source. The check is targeted: with the defining sentence present
but the edit two steps away, following returns to 0.958, equal to the
deep condition.

Claude Sonnet 4.5 uses more evidence. At matched trace length, adding a
sole-source checkpoint raises its reversion from 0.18 to 0.78 (bridge of
four operations). Its default trust also depends on how much trace
exists: with a one-line visible trace it rederives from the prompt at
0.90; with eight lines it follows the trace at 0.82.

Llama 3.3 70B follows edits at 0.97 in both the deep and the defined
conditions, despite the capacity to rederive (it solves the same chains
from scratch at 1.00 up to five steps). Qwen models at 4B and 7B never
revert in any condition. Checking requires capacity, and capacity does
not imply checking.

## 6. Scope

Three boundary results locate where the mechanism applies. Without
writing, models fail at one to two dependent steps (five for Llama 70B),
so any serial chain beyond that lives in the trace. State that is many
independent values rather than a chain behaves differently: a
box-tracking task with equal state size is tracked internally with a
fifth of the writing, and writing does not predict its correctness.
Ablating the top concepts a Jacobian lens reads from the activations, at
the strongest dose that leaves plain text generation unchanged, has no
effect on any of this, so the memory studied here is not carried by the
lens-readable subspace at that dose.

## 7. Related work

Expressivity theory shows that fixed-depth transformers can perform only
bounded serial computation in one forward pass and that chain-of-thought
length buys serial power (Merrill and Sabharwal 2024; Li et al. 2024;
Feng et al. 2023). This predicts that long dependency chains must live in
the generated text; it does not describe the storage or read mechanism,
which is what we test.

Our interventions build on causal tracing and activation patching
(Meng et al. 2022) and interchange interventions (Geiger et al. 2021);
the attention block is related to attention knockout in circuit analysis
and to attention-flow perturbation on reasoning traces (arXiv:2606.10646),
which flips answers by perturbing flow hubs; we add value-specific edges
with matched-token-type controls and the illegible-token dissociation.
Hidden-state patching of chain-of-thought representations appears in
arXiv:2604.23351, and Self-Notes (Lanchantin et al. 2023) studies writing
as memory behaviorally.

Faithfulness studies established that stated reasoning and internal
computation can diverge (Turpin et al. 2023; Lanham et al. 2023), and
probing work reads reasoning state from activations before or alongside
the text (arXiv:2603.01437; arXiv:2412.01113). Filler-token results show
models can compute through uninformative tokens (Pfau et al. 2024). These
results bound our claim: they concern values a model can hold or
regenerate internally, which our capacity and box-tracking results place
outside the mechanism's scope.

Closest to this work, arXiv:2606.29522 edits internal representations of
scratchpad state and finds downstream following at 0.80 to 0.91, on a
Qwen-7B fine-tuned for a synthetic task; arXiv:2505.04955 intervenes on
value-storing chain-of-thought tokens in multiplication and dynamic
programming tasks. We differ in testing unmodified pretrained models
across three families, localizing the read to attention edges with
matched controls, showing slot composition, and mapping when the written
record is checked against the prompt. The framing of the chain of thought
as external working memory appears without experiments in
arXiv:2602.15868.

## 8. Discussion and limitations

For monitoring, the mechanism gives a scoped guarantee and a scoped
warning. On serially dependent values beyond a model's internal capacity,
the trace is the computation's memory: what is written is what is used,
and this becomes more true as models get stronger. On values a model can
regenerate, and for models trained to check, the trace can silently
diverge from the computation.

Limitations: tasks are arithmetic and narrative-count chains, chosen for
exact ground truth; richer state (code, multi-entity plans) is untested;
the white-box evidence is at 4B to 14B; frontier evidence is behavioral;
the verification-policy differences are described for four models and are
not a law; the patch overwrites the full residual band rather than an
isolated value subspace.
