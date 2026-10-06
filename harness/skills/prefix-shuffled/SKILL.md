---
name: prefix-shuffled
description: Predicts the single most likely next scientific/engineering decision for a research project observed at a natural early stage (one, two, or three recorded decisions), using only the objects, operations, categories, and protocol vocabulary present in the observable prefix.
---

# Early-prefix next-decision prediction

## Scope

The history you see is complete: the project has genuinely made only one, two, or three
decisions. Do not posit missing earlier decisions, retained cores, supersession chains, or
project outcomes. Predict one decision: a **category** (method / experiment / ablation) plus a
one- or two-sentence statement of **which object is acted on** and **what operation is applied**.

## Step 1 — Inventory the prefix

For each observed decision extract:

1. **Object** — the concrete artifact touched (loss/objective, backbone, scorer or judge,
   probe/head, data generator, dataset construction rule, accounting/calibration routine,
   evaluation protocol, metric, hyperparameter grid).
2. **Pipeline slot** — data construction → model/objective → scoring/estimation →
   evaluation protocol/metrics → sweeps.
3. **Operation** — define, extend, replace, operationalize, sweep, measure, compare.
4. **Granularity** — config-level (explicit shapes, rates, thresholds) vs. conceptual
   (formulations, assumptions, protocol semantics).
5. **Outcome flag** on the *latest* decision (retained / abandoned / superseded). Use it only
   as described in Step 3.

## Step 2 — Category prior

Experiment is the modal next category at every prefix length, and its share grows with prefix
length; ablation is the rarest. Refine with these observed transitions:

**From one decision**
- Latest is *experiment* → most often another *experiment*; otherwise *method* or *ablation*.
- Latest is *method* → most often another *method*; otherwise *ablation* (a sweep over the new
  method's own knob).
- Latest is *ablation* → almost never another ablation; predict *experiment*, else *method*.

**From two decisions**
- Homogeneous prefix (both experiment, both method) → continuing the same category is likely,
  but a switch to the complementary category (method after two protocol decisions; experiment
  after two method decisions) is nearly as likely — decide with Step 3.
- (method, ablation) → *method* (return to method building; the ablation was a one-off probe).
- (ablation, experiment) → *experiment*.
- (experiment, ablation) → *ablation* or *experiment*; prefer *ablation* when the prefix is
  config/sweep-flavored and the first ablation opened a design space with obvious sibling
  variants; prefer *experiment* otherwise.
- (ablation, method) → *ablation*.

**From three decisions**
- If the prefix alternates A–B–A, continue the alternation (predict B).
- Long homogeneous experiment runs usually continue as *experiment*; a switch to *method* occurs
  when the experiments so far only fixed protocol/measurement and no objective or core
  procedure has been stated yet.
- Long homogeneous method runs continue as *method* while pipeline slots remain empty, and
  switch to *experiment* (baseline comparison or dataset-level evaluation) once the pipeline is
  essentially specified.
- After ablation(s) that were exploratory, expect *experiment* or *method*, not further ablation.

## Step 3 — Choose the object

Object continuity is the strongest predictable signal. Ranked rules:

1. **Retained latest method decision → act on that same object.** Extend it with an extra
   branch/signal/variant, ablate one of its components, or state the variant that differs in one
   design choice (e.g. include vs. exclude a subset, uniform vs. skewed initialization).
2. **Abandoned or superseded latest decision → shift object.** The next decision typically moves
   to a different artifact in the pipeline rather than refining the discarded one; a shrunken or
   simplified re-attempt of the discarded object is a secondary possibility.
3. **Unfilled pipeline slot.** If the prefix specifies evaluation protocol/metrics but no
   objective, predict the training objective or core procedure. If it specifies an objective or
   preprocessing but no architecture/backbone, predict the backbone. If it specifies a method
   but no data source, predict dataset construction or a simulation generator.
4. **Provisional implementation flagged in the prefix.** Vocabulary such as rule-based,
   conceptual/formula-only, temporary, default, standard, or a gradient-trained auxiliary stage
   predicts a *replacement/operationalization* of exactly that component by a deterministic,
   executable, or closed-form alternative.
5. **Parallel condition not yet covered.** If the prefix fixed a protocol on one dataset, model
   scale, adjacency/split convention, or task format, predict the same protocol transferred to a
   second dataset/scale/task format, holding the accounting or metric fixed.

## Step 4 — Operation templates

Pick the template matching Step 3:

- *Extend*: add a second branch, an extra signal family, or an additional diagnostic to the
  existing method and make it the default.
- *Replace*: swap a provisional component for a deterministic/closed-form/simpler one, or swap a
  configuration for a smaller/larger counterpart.
- *Operationalize*: turn a stated quantity into a computable score with explicit eligibility
  filters and exclusions.
- *Transfer-evaluate*: run the established protocol on a new dataset, scale, or task
  formulation, comparing against the same baseline/calibration.
- *Protocol-harden*: add repetitions/seeds, split conventions, aggregation rules, or judge
  reliability checks.
- *Sweep*: vary one declared axis (component subset, count, threshold pair, learning rate,
  scale cutoff) and report the prefix's existing metric.

## Step 5 — Specifications are underdetermined

Do not invent: numeric hyperparameters, seed values, dataset or model names absent from the
prefix, layer counts, thresholds, sweep endpoints, or benchmark suites. If specifics are needed
for coherence, reuse values or names already in the prefix, or describe them functionally
("the same metric", "a second dataset of the same type", "one design axis"). Also unpredictable
this early: which decision will later be retained or abandoned, and the project's headline
claim. Match the prefix's granularity: config-heavy prefixes get a config-level prediction,
conceptual prefixes get a formulation-level prediction.

## Step 6 — Fallback for ambiguity

When Steps 2–3 conflict or no object stands out, predict an **experiment** that evaluates the
most recently introduced object using the protocol, metric, and baseline already named in the
prefix, extended to one additional condition (dataset, scale, or task format). This is the
highest-frequency safe answer and rarely misses badly on category.

## Failure modes to avoid

- Predicting an ablation immediately after a single ablation.
- Refining an object whose latest decision is marked abandoned/superseded.
- Jumping to headline benchmark results or write-up decisions; early next steps stay inside the
  pipeline being assembled.
- Emitting invented numbers, which cost accuracy without improving object/operation match.
