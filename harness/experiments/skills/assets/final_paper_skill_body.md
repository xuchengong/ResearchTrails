# Predicting the Next Decision from a Short Prefix

## What the evidence is and is not

The evidence base is a set of final published papers. A paper's exposition order is
rhetorical, not chronological. Therefore: never claim to know what the authors did next,
never invent a supersession chain, and never treat an ablation table as a record of failed
attempts. What a final paper *does* reliably reveal is **structural dependency**: which
artifacts must exist before other artifacts can exist, and which measurements are forced
once a claim is made. Structural dependency is the only signal that transfers to prefix
prediction. Phrase every prediction as "the next decision most plausibly instantiates X,"
not "the authors then did X."

## Step 1: Classify the prefix

Read the prefix and assign each observed decision to one slot. Recurring slots across the
papers are:

- **Target object** — the entity being studied, improved, measured, or generated (a loss
  function, a class of models, a latent representation, a dataset's labels, a physical
  quantity).
- **Framing move** — a reinterpretation, reformulation, or reduction that changes which
  tools become applicable (binary classification recast as OOD detection; CRL recast as a
  measurement model; DP parameters recast as hypothesis-testing trade-off curves;
  brainteaser-solving recast as a creativity-vs-brute-force contrast).
- **Mechanism** — the concrete intervention: a loss term, a module, an encoder, a
  reweighting scheme, a pruning rule, a parameterization.
- **Substrate** — dataset, benchmark, simulator, model family, or backbone.
- **Measurement** — a metric, score, statistical test, or evaluation protocol.
- **Comparison structure** — baselines, ablations, sweeps, held-out regimes.

Then record the prefix length, since the reliable inference differs sharply by length.

## Step 2: Length-conditioned inference

**One decision observed.** Almost nothing about the eventual mechanism is inferable. What
*is* inferable is the immediate structural gap. A lone target object with no substrate
implies the next decision most plausibly fixes a substrate or a measurement; a lone framing
move implies the next decision instantiates the framing as a concrete mechanism or
quantity; a lone substrate implies the next decision names what property of it will be
studied. Prefer the decision that makes the first decision *evaluable*. Do not predict a
specific architecture, loss, or hyperparameter at this length.

**Two decisions observed.** The pair usually fixes an axis. Identify what the two decisions
jointly constrain and what they leave open. If the pair is (framing, mechanism), the open
slot is measurement or substrate. If the pair is (object, substrate), the open slot is the
mechanism or the property being measured. Across the papers, a framing move paired with a
mechanism is repeatedly followed by a metric or test purpose-built to make the framing
falsifiable — a novel score, a distance, an error measure — because generic metrics do not
express the reframed claim. This is the single strongest two-decision regularity.

**Three decisions observed.** The design is largely specified and the next decision is
usually *validation structure* rather than new machinery: the baseline set, the ablation
that isolates the newest component, or the stress regime (held-out model, unseen domain,
adversarial perturbation, different language, different compression rate). At this length,
predicting a further new mechanism is usually wrong; predicting the evaluation that would
be required to support the assembled design is usually right.

## Step 3: Recurring structural dependencies

These relationships appear in multiple papers and are the core transferable content.

1. **A new claim about a property forces a new measurement of that property.** When a
   project asserts something a standard metric cannot express, a bespoke measurement
   follows. Observed repeatedly: a flux-consistency claim accompanied by a flux-error
   measure; a specification-alignment claim accompanied by an exclusivity score built from
   conditional independence tests; an operational-risk claim accompanied by an
   attack-success-versus-baseline quantity; a creativity claim accompanied by a
   step-level creative-versus-rudimentary annotation; a reasoning-step-importance claim
   accompanied by a per-step importance score.

2. **A mechanism that modifies training implies a frozen-backbone or unchanged-inference
   variant, and vice versa.** Projects that introduce a training-side intervention commonly
   also expose a cheap post-hoc or plug-and-play form, and projects that start post-hoc
   commonly add a trainable version. If the prefix shows one side, the other side is a
   plausible next decision.

3. **An intervention with a free scalar or weight implies an explicit study of that
   scalar.** Whenever a prefix introduces a knob — a mixing coefficient, a pruning ratio, a
   compression rate, a loss weight, a rank, a temperature/bias — a sweep or sensitivity
   analysis over it is a high-probability next decision, often with a monotone-degradation
   or sweet-spot finding.

4. **Two candidate regimes imply a controlled contrast holding everything else fixed.** When
   a prefix sets up specialized-versus-generic, pixel-versus-latent,
   generative-versus-deterministic, or in-domain-versus-out-of-domain, the next decision
   commonly fixes the shared backbone, budget, or protocol so the contrast is attributable.

5. **A pipeline with several stages implies per-stage diagnosis.** If the prefix commits to a
   multi-stage procedure (encode then generate; parse then prove; segment then score then
   prune; propose then verify), the next decision plausibly makes each stage independently
   checkable so failures can be localized.

6. **Scale or diversity of the substrate is itself a decision.** Several papers show that
   conclusions depend on how many sources, models, languages, or time points are included,
   and follow with an explicit study of that dependence. If the prefix fixes a narrow
   substrate, broadening it — or measuring the effect of breadth — is plausible.

7. **A human or expert-in-the-loop component implies agreement and cost accounting.** When a
   prefix involves annotation or judging, the next decision commonly measures inter-rater
   agreement, validates an automatic judge against humans, or reports effort saved.

8. **A theoretical or formal argument implies an empirical instantiation, and an empirical
   pattern implies a formal characterization.** Prefixes that open with an impossibility,
   bound, or reformulation commonly proceed to a concrete algorithm; prefixes that open with
   a surprising measurement commonly proceed to a condition or argument that explains it.

## Step 4: Produce one prediction

Output exactly one next decision, composed of an **object** and an **operation**.

- Choose the object from the slot the prefix leaves structurally open, using Step 2.
- Choose the operation from the dependency in Step 3 whose precondition the prefix most
  clearly satisfies. If several apply, prefer the one that makes an existing claim in the
  prefix checkable over the one that adds new capability.
- Keep the specification coarse. Name the *kind* of artifact (a bespoke consistency metric,
  a frozen-backbone variant, a sweep over the introduced coefficient, a controlled
  equal-budget contrast), not its numeric or architectural details.

## Step 5: State what is not inferable

Always list, briefly, the things the prefix does not determine. Typically: exact
architectures and layer counts; optimizer, learning rate, batch size, epochs; dataset sizes
and splits; thresholds and coefficient values; the number of baselines; which specific
ablation cell will be reported; and whether any particular result will be positive. Never
fill these in from prior familiarity with a published paper.

## Conservative fallback

When the prefix is genuinely ambiguous — the slots are unclear, or three or more
dependencies apply with equal force — predict the **minimal evaluability decision**: fix the
measurement or protocol that would let the already-observed decisions be assessed. This is
the lowest-variance choice because every paper in the evidence base must eventually
establish how its central claim is judged, whereas mechanism choices vary widely. Prefer
this over guessing a mechanism, and prefer a single concrete measurement decision over a
hedged list.

## Failure modes to avoid

- Predicting a chain of several future decisions instead of one.
- Naming numeric hyperparameters, specific model names, or specific datasets.
- Asserting that an alternative was tried and abandoned.
- Predicting a novel mechanism at prefix length three, where validation is far more likely.
- Predicting a metric at prefix length one, when no claim yet exists that needs measuring —
  unless the single decision is a framing move whose whole point is a new quantity.
- Recycling generic research advice ("do a literature review," "tune hyperparameters") that
  none of the evidence papers records as a decision.
