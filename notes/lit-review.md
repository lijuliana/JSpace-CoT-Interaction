# Literature review

Four sweeps: latent reasoning, CoT faithfulness and monitorability, mech interp methods, and difficulty scaling / task design. This file is the synthesis; the goal is to know exactly what is already established, what is contested, and where the open ground is.

## 1. What is already established

**CoT tokens can serve as externalized computation state.** This is settled at three levels. Theory: fixed-depth transformers are limited to roughly TC0 in a single forward pass (Merrill and Sabharwal 2023, arXiv:2207.00729), and CoT length buys computational class in a precise way, with linear steps reaching regular languages and polynomial steps reaching P (Merrill and Sabharwal, ICLR 2024, arXiv:2310.07923; Li, Liu, Zhou, Ma, ICLR 2024, arXiv:2402.12875; Feng et al., NeurIPS 2023, arXiv:2305.15408, where CoT tokens literally hold DP table entries). Mechanism, in toy models: the "iteration head" circuit (Cabannes et al., NeurIPS 2024, arXiv:2406.02128) implements an RNN whose hidden state is the emitted token stream. Framing: Korbak et al. 2025 (arXiv:2507.11473) state the necessity argument for safety, that hard serial tasks force CoT to act as working memory. So "CoT is external memory" is background, not a contribution.

**CoT is not always load-bearing.** Turpin et al. 2023 (arXiv:2305.04388) showed rationalization; Lanham et al. 2023 (arXiv:2307.13702) showed with truncation and corruption tests that CoT reliance varies by task and, importantly, decreases with model capability on a fixed task. Bentham et al. 2024 (arXiv:2402.13950) showed that measured "faithfulness" largely tracks whether the model can already do the task internally. Anthropic 2025 (arXiv:2505.05410) extended this to RL-trained reasoners: hint reveal rates under 20 percent. The capacity confound in the faithfulness literature is actually our object of study.

**A candidate internal workspace has been located and named.** The Jacobian-lens line of work identifies a small set of residual directions, the J-space, whose top-k active vectors behave like a latent workspace band: ablating the residual projection onto them at each position leaves parsing, classification, and one-step recall intact but collapses internal multi-step reasoning, and GSM8K solved with explicit CoT is far more robust to the ablation than the same problems answered directly. That asymmetry is the origin of the internal-external interchangeability question this project studies. Two cautions from the source material itself: the ablation is defined over a fitted lens artifact (the public neuronpedia/jacobian-lens release), so claims inherit the artifact's quality, and the robustness contrast is at fixed task difficulty, leaving the difficulty dependence unmeasured. We treat J-space as one operationalization of the workspace to be tested against, not assumed; our probe- and lesion-derived workspace definitions are the independent check.

**Substantial multi-step computation happens in activations.** Anthropic's attribution graph work (transformer-circuits.pub, 2025) found forward planning at newline tokens and latent two-hop recall in a frontier model. Latent multi-hop is real but unreliable and degrades with hop count (Yang et al., ACL 2024, arXiv:2402.16837), and it fails for a mechanistically specific reason: the first hop sometimes resolves in layers too late for the second hop to use, and patching earlier-layer states repairs many failures (Biran et al., EMNLP 2024, arXiv:2406.12775). That last result matters a lot for us: layer depth is a serial step budget, and running out of layers is a concrete, observable form of workspace pressure.

**Task difficulty modulates the internal/external split behaviorally.** Kudo et al. 2024 (arXiv:2412.01113) found single-step answers are decided before CoT begins and multi-step answers are decided during CoT emission. Emmons et al. 2025 (arXiv:2507.05246) found models cannot evade CoT monitors when tasks are hard enough that CoT is computationally necessary. The Apple "Illusion of Thinking" exchange (arXiv:2506.06941 and rebuttals, especially arXiv:2506.09250 and 2507.01231) is a cautionary tale: their high-difficulty collapse partly reflected an output-length confound, which is why our difficulty knobs must decouple required output length from required serial computation.

**Latent-reasoning alternatives exist and have known character.** Coconut (arXiv:2412.06769) and the superposition theory around it (arXiv:2505.12514) show continuous latent thought carries superposed parallel candidates, which discrete tokens cannot. Pause and filler tokens (arXiv:2310.02226, arXiv:2404.15758) add parallel compute width, not serial depth; Pfau et al. is the cleanest existing dissection of compute-slots vs content-bearing memory, and filler substitution is our key control condition.

## 2. What is contested or missing

- Nobody intervenes on the model's internal capacity and shows computation migrating into the token stream. The exchange rate between depth and CoT length exists as a theorem (Li et al. 2024) and as a behavioral correlate (Lanham et al.), never as a measured mechanistic quantity.
- No study of write/read-back dynamics: whether internal representations of an intermediate value decay after the value is written to tokens (eviction), and whether models refill internal state by attending back to their own CoT. Iteration heads are the nearest thing, in toys only.
- No fitted quantitative law for when models start writing intermediate results down. Closest: Snell et al. difficulty-binned compute allocation (arXiv:2408.03314), optimal-CoT-length scaling fits (arXiv:2504.01935), and Apple's descriptive three-regime plot. None fit a crossover threshold as a function of difficulty and model scale.
- Nearly all mechanistic evidence is GPT-2 scale or from-scratch toys. Interp on R1-style RL-trained reasoners is thin: SAE feature work (arXiv:2503.18878, Goodfire's R1 SAEs) and steering vectors for reasoning moves (Venhoff et al., arXiv:2506.18167) exist, but no workspace-level causal analysis.
- No information accounting across the boundary: how many bits of task-relevant intermediate state sit in activations vs tokens at each point in a trace.
- The J-space ablation asymmetry (CoT robust, direct fragile) has not been tested across difficulty, and it is not known whether the J-lens directions carry the specific intermediate values that written tokens carry, or a more diffuse capability. The construct and the value-level read-back mechanism have never been connected in either direction.

## 3. Methods we will build on

From the interp methods sweep, the toolkit and its known failure modes:

- Causal claims come from patching, not probes or SAEs. Counterfactual (resample) patching with logit-diff metrics is the safest default (Zhang and Nanda, arXiv:2309.16042; Heimersheim and Nanda, arXiv:2404.15255). Zero and mean ablation take the model off-distribution and overestimate importance (Li et al., NeurIPS 2024 optimal ablation).
- Subspace-level interventions need the Makelov illusion check (arXiv:2311.17030); dormant pathways can fake causal alignment. DAS (arXiv:2303.02536) finds subspaces rather than assuming a basis.
- Probes require Hewitt-style control tasks and selectivity reporting. SAE-based claims need linear probe baselines: SAE probing loses to logistic regression across regimes (Engels et al., arXiv:2502.16681), and SAEs do not find canonical units (Leask et al., ICLR 2025). We use SAEs for discovery only.
- Attribution patching (arXiv:2310.10348, AtP* arXiv:2403.00745) makes screening thousands of sites affordable at 7B to 32B scale before exact patching confirms.
- Sentence-level counterfactual resampling on reasoning traces (thought anchors, Bogdan et al., arXiv:2506.19143) is the state of the art for which written sentences are causally read back, including the "receiver head" mechanism. This is the external-side tool to pair with our internal-side interventions.
- The controls any ablation claim needs: random directions of matched rank and norm, control tasks that must stay flat, dose-response curves, and a global damage meter (KL to the clean model).
- Subspace ablations built on fitted dictionaries (J-lens, SAEs) add two requirements: dose calibration on neutral text before any task outcome is seen (a strength gate on perplexity and top-1 agreement against the clean model), and controls drawn from the same dictionary rather than isotropic random directions, since dictionary vectors are not orthogonal and joint projection changes the removed norm. Full-strength projection removal is generally off-distribution in the same way zero ablation is.

## 4. Task families

From the difficulty sweep, the families with a clean scalar knob where low difficulty is solvable without CoT:

- modular arithmetic chains (k operations; TC0 boundary understood; grokking-compatible)
- k-hop reachability / variable chains, LEGO style (arXiv:2206.04301); theory predicts an internal ceiling near log-depth in the layer count (Sanford et al., NeurIPS 2024)
- entity/box tracking (arXiv:2305.02363); load = entities times updates; known circuits
- iGSM-style synthetic math (Ye et al., arXiv:2407.20311), which comes with probing evidence that models precompute dependency graphs internally

Tower of Hanoi and similar planning tasks only as stress tests with program-output controls, because of the output-length confound.

## 5. Novelty position

The claim "models externalize when internal capacity is pressured" exists as theory and as scattered behavioral correlates. What does not exist, and what this project targets:

1. causal migration: manipulate internal capacity (or the external channel) inside a fixed model and watch computation move across the boundary, with circuit-level evidence
2. eviction and read-back dynamics of the hierarchy, measured with probes plus patching
3. a fitted crossover law for externalization onset across difficulty and a model-size ladder, tested against the log-depth prediction from theory, and checked across model families

A 2025-26 wave now covers much of the correlational ground: answers decodable from activations before verbalization (arXiv:2603.05488 "Reasoning Theater", arXiv:2603.01437), activations carrying more state than the trace shows (arXiv:2604.18307, arXiv:2606.13603), and depth-parameterized state-tracking probes (arXiv:2603.27343, WMF-AM). This is why observation-only phases are framed as calibration and the contribution rests on the interventions.

Nearest neighbors to cite and position against: Li et al. 2024 (the theorem), Kudo et al. 2024 (the behavioral dissociation), Lanham et al. 2023 (capacity-dependent reliance), Korbak et al. 2025 (the framing), Bogdan et al. 2025 (the external-side tooling), and the Jacobian-lens paper (the workspace construct and the ablation asymmetry this project starts from). Against the last, the project cuts both ways: our synthetic families supply the difficulty axis and direct-answer headroom its GSM8K contrast lacks, and the value-level patch decomposition asks whether the J-lens directions carry the specific quantity that read-back uses, which tests the construct rather than assuming it.

The two closest recent items, read in full:

- arXiv:2604.15726 ("LLM Reasoning Is Latent, Not the Chain of Thought") is the direct antagonist. Its headline rests on aggregate mediator contrasts (ablating a latent state hurts more than corrupting the surface trace, in an "ordinary" task regime), and its own regime table concedes surface traces win when traces are constitutive. It corrupts the surface only as an aggregate contrast, never token-level corruption of specific intermediate values with downstream tracking, and its section 4.2 explicitly calls for exactly that design. Our gate B and 2c are that design. If downstream computation tracks corrupted written values and internal copies decay after writing, CoT is a load-bearing store under pressure, which bounds the generality of their H1.
- arXiv:2605.30343 (RiM, "Unlocking the Working Memory of LLMs") is the constructive counterpoint, and compatible with our thesis: fixed latent memory blocks widen internal capacity so writing becomes unnecessary at small scale (Llama-3.2-1B, GSM8K). It never manipulates capacity or difficulty and never measures eviction. Cite its block-specific workspace representations as convergent evidence for the working-memory framing.

Neither paper does any of: corrupting written intermediates with downstream tracking, capacity lesions measured against externalization, pre/post-write decodability, or an onset law. Verified against full texts 2026-07-31; spot-check PDFs before citing specific numbers.
