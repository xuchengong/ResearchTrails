# Research Decision Prediction

## Output contract

Predict exactly one decision and state four things: its **type** (method / experiment / ablation / hypothesis), the **single component** it acts on, the **relation** to existing decisions (introduce a new stage, replace one stage in place, extend coverage, fix a default, demote a variant to a control, discontinue a branch), and enough **protocol detail borrowed from the project's own conventions** to be checkable (its existing sweep style, seed counts, metrics, data sources). Predictions that name a component not already present in the trajectory, or that bundle several changes, are almost always wrong.

## Step 1 — Read the trajectory as a set of open debts

Decision order, not identifier order, carries the signal. Build three lists:

- **Open refinement debts**: components introduced in a conceptual, hand-set, approximate, or workaround form. Anything defined as a formula rather than an executable quantity, any hand-chosen threshold or capacity, any "exclude the hard cases" filter, any free-form generative stage asked to preserve content, any multi-step construction that approximates a quantity obtainable directly.
- **Stabilized core**: retained method stages that later decisions already build on.
- **Peripheral branches**: arms with their own training loop, their own simulation, or their own data path not shared with the core.

The next decision overwhelmingly lands on the newest open debt, or extends the stabilized core; it rarely opens an unrelated topic.

## Step 2 — If a debt is open, predict a same-component replacement

Replacement, not addition, is the default prediction, and its direction is predictable:

- Conceptual or idealized definition → **operational surrogate computed from quantities the system already emits** (likelihood/perplexity of the final output, stored parameters, distances already logged), with explicit skip rules for degenerate cases.
- Chain of approximations (pointwise minima, grids, hulls, intermediate reconstruction stages) → **one direct computation over the same primitives**, with edge cases handled explicitly.
- Soft penalty/regularizer enforcing a constraint → **hard enforcement moved outside the main computation** (or the reverse). When one regime fails, predict the opposite regime for the same constraint, not a new mechanism.
- Hand-set capacity, count, or density → **value derived from the data itself** within the same relaxation.
- Filter that drops difficult items → **procedure that processes them in pieces**; if that also fails, formal exclusion recorded as declared scope.
- Free-form rewriting/selection by a model → **structured, staged annotation** that first partitions, then labels/links, then acts, so the acted-on units have a stated role.
- Judge or annotation protocol → **richer inputs first** (add sampled system responses, then their graded outcomes, then more samples), **then a stronger backend model**; only later is it generalized, and generalization usually fails.
- A component gaining a second branch or mixture → **a fixed default mixing value, with the endpoints exposed so they can be ablated separately**.
- Strong qualitative or sign-based hypothesis → **abandoned in favor of a continuous measured quantity plus an explicit parameter** that governs it.

When two variants of one component coexist, predict either an ablation comparing their composition/order or a decision that fixes one as default — not a third variant.

## Step 3 — If the core is stable, predict the next expansion move

These follow a reliable order; pick the earliest one not yet done:

1. **Same protocol, another source of the same kind**: a second or third data family, base model family, backbone depth, or task variant, each as its own decision. A coverage addition that does not match the claim gets replaced by a different paradigm rather than dropped silently.
2. **Matched final protocol**: unmodified controls versus treated variants across all accumulated families under one grading rule, with deterministic/rule-based grading preferred over model-judge grading for headline numbers.
3. **Sensitivity ablations on retained knobs**: strength/ratio/coefficient sweeps, component-order, regularization strength, how much of the stabilized pipeline is used, or the degenerate/random version of the method as a control. Variants earlier proposed as methods reappear here as controls.
4. **Reporting and aggregation rules**: repeat each configuration and report mean and dispersion; aggregate per-condition before overall; exclude degenerate conditions; report normalized and absolute gains; add a cost-normalized metric (per-token, per-unit-error, runtime) beside quality; report preferred settings only where the treatment beats the control.
5. **Artifact checks on the headline effect**: many seeds or initializations, repeated judgments, or tracking the quantity during training rather than only at convergence.

For simulation- or data-generation-based projects, the recurring move is **enlarging the generator so it contains the variables the downstream analysis needs**, or **appending one more physically/statistically motivated stage or correction** while preserving previously added constraints — restating the full generative specification each time.

## Step 4 — Predict discontinuation when the cues fit

Predict abandonment or removal (rather than refinement) when a branch: needs a training loop or pipeline no other arm uses; lives in its own simulation or benchmark disconnected from the main data path; is an exhaustive combinatorial or threshold sweep over a component already fixed; adds a baseline whose conclusion the retained results already imply; extends an expensive labeling protocol to many additional benchmarks; or cheapens/distills a component that is not the project's claim. Consolidation toward one pipeline and a small fixed set of signals is the steady late-stage tendency.

## Tie-breaking priors

- Prefer the smallest edit that removes a stated fragility over any broadening of scope.
- Prefer acting on the component touched most recently; debts introduced long ago and still open are resolved late, usually by deletion or by declared scope limits.
- Type transitions: method → method (refinement) → experiment (apply to new source) → ablation (knob sensitivity) → experiment (matched protocol and reporting). Early trajectories are method-dense; late ones are evaluation- and aggregation-dense.
- Keep the granularity of the records: one stage, one factor, one reporting rule per decision.

## Failure modes to avoid

Do not invent mechanisms absent from the trajectory; do not predict a return to an already-superseded form; do not predict broad generalization while an open debt remains in the core; do not predict a metric or judge unrelated to the quantities the project already computes; do not predict "run more baselines" when the project is visibly narrowing.
