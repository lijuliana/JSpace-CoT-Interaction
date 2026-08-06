# Experimental plan

Question: when does a reasoning model hold intermediate state in activations vs write it into tokens, what mechanism performs the handoff, and is the crossover predictable?

The memory hierarchy hypothesis, stated so it can fail: the residual stream is a fast workspace whose serial capacity is bounded by depth and whose parallel capacity is bounded by superposition interference; the token stream is a slow, durable store written one discrete symbol at a time. Models allocate intermediate state between the two tiers based on load, write values out when internal capacity is pressured, release the internal copy afterward, and read values back through attention when needed. Each clause of that sentence is a separately testable claim, and several could be false while others hold. The design below tests them separately.

Terminology: "workspace" means the task-relevant subspace of the residual stream at reasoning-time positions, located empirically (phase 2), not assumed.

## Models

- primary ladder: DeepSeek-R1-Distill-Qwen 1.5B / 7B / 14B / 32B. Actual reasoning models, one family, four sizes. All white-box work here.
- counterparts: Qwen2.5-Math / Qwen2.5-Instruct at matched sizes, to separate "reasoning-trained" from "big".
- generalization check: Llama-3.1-8B / 70B (and R1-Distill-Llama-8B/70B), second architecture family.
- Gemma-2-9B with Gemma Scope SAEs for cheap feature-level discovery passes, since public SAEs exist.
- API models (Bedrock, Anthropic) for behavioral sweeps only.

## Task families

Requirements: a scalar difficulty knob d, difficulty decoupled from required output length, low-d instances solvable with no CoT, intermediate values that are exactly specifiable so probes and corruptions have ground truth.

1. modular arithmetic chains: k operations mod p. d = k.
2. variable chains (LEGO style): a=5; b=a+2; c=b*3; ... query a late variable. d = chain length; distractor variables control parallel load separately from serial depth.
3. entity tracking: n boxes, m moves, query final contents. d = (n, m); this knob stresses parallel storage more than serial depth, deliberately complementary to family 2.
4. k-hop reachability on random DAGs. d = hop count. Theory (log-depth results) makes a quantitative prediction here.

Two knobs on purpose: serial depth (families 1, 2, 4) vs parallel storage load (family 3 and distractor count). The hierarchy hypothesis says both create pressure; the depth-only alternative (running out of layers) says only serial depth does. This is one of the places the design can distinguish hypotheses.

All generators seeded, instance-deduplicated, with held-out difficulty levels for extrapolation tests.

## Phase 0.5: gate experiments (week 1, before anything expensive)

Two cheap pilots run before probe training or large sweeps, because their outcomes decide the paper's framing.

- gate A (does the write policy respond online at all): crudest internal squeeze, a layer-window resample lesion at one site on R1-distill-7B, one task family, 200 instances, checking only whether externalization fraction moves relative to random-site lesions at matched damage. Distilled models imitate teacher traces, so their write policy may be entirely open-loop; if this gate is flat, phase 3a is demoted from headline to section and the paper leads with eviction, read-back, and the onset law, which stand without it.
- gate B (is anything read back): token-corrupt written intermediate values and measure answer flip rate vs d. The antagonist position ("reasoning is latent, the CoT is a projection") predicts flip rate near zero at all d. This pilot tells us which paper we are writing before we commit.

Sampling config, fixed here up front: temperature 0.6 top-p 0.95 for R1-distills (their recommended settings), greedy for probing runs where determinism matters, with a sensitivity check between the two, since trace structure shifts with sampling and "CoT length increased" is noisy at temperature.

## Phase 1: establish the phenomenon (behavioral, cheap, mostly API + small GPU)

For each model and task family, sweep d and measure under three conditions: forced direct answer (no CoT), free generation, forced CoT. Record accuracy, CoT length, and which intermediate values appear verbatim in the trace (exact matching against ground truth, which our synthetic tasks make possible).

Key quantity: the externalization curve, fraction of ground-truth intermediate values written down, as a function of d. The hypothesis predicts a characteristic shape: near zero below some d*, rising afterward, with d* increasing in model size. See hypotheses.md before any run.

Measuring externalization: exact match is the primary detector but is unsound alone, since values appear in variant surface forms and tokenizers split numbers differently across families, which would contaminate cross-family d* comparisons with verbalization style. So: (a) a programmatic normalization layer canonicalizes numerals and simple expressions (no LLM judging), and (b) on a subsample we use the causal definition, v_i counts as externalized iff corrupting its written form changes the answer. We report agreement between surface and causal definitions; where they diverge the causal one wins. A value can also be externalized as a derived quantity without appearing verbatim, which only the causal definition catches.

One confound handled by design: forcing a direct answer on an R1-distill means suppressing the think block, which is off-distribution by different amounts across the ladder, so d* would partly measure template robustness. Direct-answer capability estimates therefore come primarily from the matched Qwen2.5 base/instruct models; R1-distill no-think numbers are reported as secondary. The behavioral ladder also picks up Qwen 0.5B and 3B sizes, nearly free, to double the points available to phase 4 fits.

Framing note: recent work already establishes the correlational premise that activations carry answers and state beyond what the trace shows (Reasoning Theater arXiv:2603.05488, pre-CoT answer decoding arXiv:2603.01437, arXiv:2604.18307, arXiv:2606.13603, WMF-AM arXiv:2603.27343). Phase 1 is calibration for the intervention phases, cited as such, and is not a contribution on its own.

Design point: also measure direct-answer accuracy at each d. If externalization onset simply tracks the point where direct answering fails, that is consistent with the hypothesis, but if models externalize far below that point (as overthinking work suggests) or far above it, the simple cost-benefit story is wrong and we need to say so.

## Phase 1.5: the protection experiment (early, reuses gate A machinery and phase 1 conditions)

The direct factorial at the heart of the question: accuracy under internal lesion x answer condition (direct vs CoT) x difficulty d. If written tokens carry state that would otherwise live in the residual stream, CoT should protect accuracy against internal lesions, and the size of that protection should depend on d in a way that discriminates hypotheses. Three outcome patterns written down in advance (see hypotheses.md): protection roughly constant in d; protection shrinking at high d (externalization is partial, the hardest problems still lean on internal state); protection growing with d (easy problems write nothing down, so there is nothing external to fall back on).

The stronger per-instance version: within a fixed (lesion, d) cell, regress lesioned-accuracy on the instance's own externalization fraction from its clean trace. Protection tracking externalization at the instance level is much harder to explain away than a cell-level correlation, and the same regression run on the random-window control arm gives the matched-damage comparison for free.

2a. probes for intermediate values. Train linear probes for each intermediate value v_i at each layer and token position, with Hewitt-style control tasks (shuffled-label probes) and selectivity reporting. This gives a map of where and when each value is represented internally. Establish the probe baseline before any SAE work; SAEs (Gemma Scope on Gemma-2, Goodfire on R1 if usable) are discovery aids only.

2b. eviction test. Track probe decodability of v_i across the trace timeline. Hypothesis: decodability of v_i at current-position residual streams drops after the token where v_i is written out, relative to cases where it is not written. The critical design point: whether the model writes v_i is endogenous (harder instances get written, and a model that chose to write is in a different state), so comparing traces by the model's own choice is a collider. Writing is therefore made exogenous: on identical instances, constrained decoding either forces externalization of v_i or suppresses it, and decodability is compared at matched distance-from-computation on the same prompts. Probing positions are content-anchored (the token after each equals sign, the sentence boundary after each step), never raw position indices, since token counts drift across formats and model sizes. Opposite result is fully visible: decodability persists or rises after writing, which would mean tokens are a broadcast copy, not an eviction target, and the "hierarchy" is really a redundant cache.

2c. read-back test. Corrupt a written intermediate value in the trace (edit the token, continue generation). If downstream computation dereferences external memory, the final answer should track the corrupted value. Then the mechanistic version: attention knockout from later positions to the tokens holding v_i, and path patching to find which heads carry it (receiver-head analysis in the style of thought anchors, at value granularity). Cross-check against internal state: patch the corrupted trace's residual stream with the clean value and see if the answer reverts. The interesting quantitative output is the read-back fraction, how much of the causal effect on the answer flows through the written token vs the internal path, as a function of d. The hypothesis predicts this fraction rises with d. Flat or falling is visible and would refute the load-shifting story.

2d. workspace identification. Two independent definitions, on purpose. First, DAS-style interchange interventions find the subspace carrying v_i at reasoning positions, with the Makelov illusion check. Second, a probe-derived subspace from 2a, which involves no optimization against causal effect. DAS optimizes for behavioral effect, so "ablating the DAS subspace changes behavior" is close to circular, and worse, DAS could select a direction whose ablation mimics a low-confidence high-verbalization regime rather than removing storage. Phase 3a therefore requires its compensation signature under both definitions, plus under optimization-free layer-window lesions, and the DAS subspace is frozen on a held-out task family before being tested cross-task.

## Phase 3: causal capacity interventions (the core contribution)

Two directions, because the hypothesis is about a two-way trade.

3a. squeeze internal, watch external. Interventions of increasing specificity:
   - layer-window lesions at reasoning positions (resample ablation from matched control prompts)
   - workspace-subspace ablation (the phase 2d subspace), dose-controlled by rank and by interpolation strength
   - attention knockout to recent non-CoT context (shrinks effective internal carry)
Measure: CoT length, externalization fraction, read-back fraction, accuracy. Hypothesis predicts a compensatory signature: externalization rises and accuracy partially recovers relative to matched-damage controls. The claim requires a four-gate conjunction, stated in advance: externalization up, read-back fraction up, control tasks flat, and a steeper task-vs-control dose curve than random subspaces at matched KL. Any subset short of the conjunction is reported as what it is. Only run at full scale if gate A passed.

Controls that separate targeted effect from generic degradation, all reported in every figure:
   - random subspaces of matched rank and norm, same layers, same positions
   - control tasks matched for output format that do not use the workspace content (e.g., copy tasks, single-step lookups); these must stay flat
   - dose-response: effects should scale smoothly with intervention strength; a targeted mechanism gives a different dose curve on task vs control than uniform damage does
   - KL to the clean model on neutral text as a global damage meter; report effect per unit KL
   - the discriminating signature: broad degradation makes everything worse everywhere; a memory hierarchy under pressure makes the model write more and lean on what it wrote. Increased externalization plus increased read-back fraction plus flat controls is not producible by uniform damage, and if we do not see that conjunction we say so.

3b. squeeze external, watch internal. Constrain the token channel: hard token budgets, filler-token replacement of the trace (Pfau-style, separating compute-slots from content), paraphrase rewriting (destroys steganographic content, keeps semantics), structured truncation. The paraphrase condition gets a self-paraphrase control (same content regenerated at a different seed) to bound plain distribution shift, since R1 traces have a distinctive voice and rewriting it is itself off-distribution. Measure internal load: probe decodability of intermediate values at late positions, dimensionality of the phase 2d subspace occupancy. Hypothesis predicts internal representations work harder (more values decodable internally, longer persistence) when writing is blocked, up to a capacity ceiling where accuracy breaks. The filler condition is the sharpest control: if filler tokens rescue performance as well as content tokens, the channel is compute, not memory, and the memory hierarchy framing is wrong for that task.

## Phase 4: the onset law

Fit the externalization onset d* (from phase 1 curves, defined by a fixed threshold on externalization fraction, with sensitivity analysis over threshold choice) as a function of model size, layer count, and task family, across the R1-distill ladder plus Llama family. Candidate laws to compare, chosen before fitting: d* linear in depth (serial budget story), d* logarithmic (parallel-scan story, per graph-connectivity theory), d* tracking direct-answer failure point (cost-benefit story). Model comparison by held-out difficulty levels and held-out model sizes, not fit quality alone. Honest caveat stated in the paper: four sizes per family spanning 28 to 64 layers discriminate linear from log only weakly; held-out-difficulty extrapolation is the real test, and the extra Qwen base sizes from phase 1 help. Check whether one law spans families or each family gets its own, and whether reasoning-trained models shift d* relative to matched base models (RL moving the write threshold is itself a finding, either direction).

Also look for a mechanistic transition marker near d*: does anything discontinuous happen in the phase 2 maps (probe decodability, read-back fraction, receiver-head attention mass) as d crosses d*, or is the behavioral crossover smooth underneath, Schaeffer-style? Both outcomes are informative and both are visible in the design; claim a phase transition only with a continuous-metric discontinuity, not a thresholded-metric one.

## Phase 5: format geometry (time permitting, or as the applied payoff)

Compare scratchpad formats at matched d: free prose, structured state dumps (explicit variable tables), code-like traces. Measure externalization efficiency (accuracy per written token), read-back fraction, and whether the phase 2c receiver circuitry differs by format. The hierarchy view predicts formats that make values easy to address (structured, code-like) raise the effective external capacity and shift d* upward. This is the section with direct design implications: if structured external memory measurably beats prose at equal token cost, that is actionable for reasoning-model training, and it connects to monitorability since a well-used external memory is a readable one.

## Evaluation choices

- exact-match on synthetic tasks; no LLM judging anywhere a program can grade
- every causal claim gets: resample ablation primary, zero/mean as robustness, logit-diff and prob metrics both reported
- seeds: 3 minimum per cell for generation-based numbers; probe results with train/val/test splits across instances, never positions of the same instance
- effect sizes with bootstrap CIs over instances; no bare p-values

## Compute plan and limitations

- phase 1: mostly API (Bedrock) plus 1x A100/H100 node for open models with vLLM. Cheap.
- phase 2 and 3: the bottleneck, and bigger than classic patching budgets suggest, because interventions happen during autoregressive generation with regeneration after each intervention, per instance per dose per seed. Budgeting is done in generated-tokens-under-intervention, not forward passes on fixed prompts. 7B is the workhorse, 14B confirmation only, 32B only for headline replications. Needs KV-cache-preserving intervention hooks; the nnsight/vLLM plumbing is its own engineering task with real time allocated. Attribution patching screens first, exact patching on the shortlist.
- 70B Llama runs: 8x A100/H100 node, reserved for the final cross-family check only.
- known limitations to state up front: R1-distills are distilled, not RL-trained from scratch, so RL-specific claims are limited (mitigated partly by the base-model comparison); DAS subspaces are optimization products and inherit the illusion risk even with checks; synthetic tasks trade ecological validity for ground truth, which is the right trade for causal work but caps the generality claims; probing establishes representation, not use, which is why every probe result that matters is paired with a patching result.

## Headline and fallback

Primary headline, contingent on gate A: induced internal scarcity causes compensatory externalization, with the four-gate conjunction as evidence. Pre-committed fallback if gate A is flat or 3a fails its conjunction: the eviction and read-back dynamics of the hierarchy (2b, 2c) plus the onset law (4), which stand on their own. A 3a null under passing controls is itself reported, as evidence that the write policy is set by training rather than adapted online, which bears directly on monitorability arguments that assume necessity.

## Order of operations

Gates A and B first, week 1, on whatever single GPU comes up soonest. Phase 1 alongside them (behavioral, cheap, and its curves pick the d ranges for everything else). Phase 2 on 7B once probing infrastructure is up. Phases 3 and 4 depend on 1 and 2 and on the gates. Phase 5 floats.

## J-space program, part 2 (planned, not yet run)

Part 1 (run) froze the ablation dose on neutral text and ran the anchor 2x2xd on chains. The remainder, in priority order:

1. Patch decomposition (harness written, src/harness/jspace_patch.py): in the read-back patch, restore only the J-space projection of the clean-minus-corrupt delta vs only its complement, with a random rank-matched projection control. Decides whether the readable value register lives in the lens-readable concept workspace. Replaces the full-residual patch as the headline version; full-residual stays as the sanity baseline.
2. Box tracking under J-space ablation (same harness, --family entity_tracking): the serial/parallel comparison with the targeted instrument. Prediction if J-space is the parallel workspace: boxes suffer more disproportionately than under the blunt lesion.
3. Multi-dose N2: repeat the cot cells at the two doses above the frozen one (k=16 alpha=1.0, and k=32) to test whether scarcity-induced writing appears at doses the freeze rule excluded, reporting neutral-text damage alongside so the trade-off is explicit.
4. Qwen3 extension: the artifact also covers qwen3-8b; rerun part 1 there to check the result is not a Qwen2.5 quirk.
5. Second-GPU parallelization when capacity returns; all four items are independent.
