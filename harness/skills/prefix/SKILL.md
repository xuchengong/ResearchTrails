---
name: prefix
description: Predicts the single most likely next research decision from a project's first one to three recorded decisions, using inventory-gap and continuation rules derived from early-stage decision trajectories across many projects.
---

# Early-Stage Next-Decision Prediction

Use this when the full observable history is one, two, or three decisions and the project is genuinely young. Nothing earlier is hidden; the prefix is the whole project so far. Your job is to name **one** next decision: its category (`method`, `experiment`, `ablation`), the **object** acted on, and the **operation**.

## Step 1 — Build the inventory from the prefix

Read the prefix and mark which of these exist:

- **Mechanism**: a proposed procedure, objective, loss, estimator, formulation, or model component.
- **Data**: a dataset, corpus, benchmark, or simulation/data-generation setup.
- **Measurement**: a metric, score, judge, diagnostic, or scoring protocol.
- **Tunables**: any parameter, threshold, temperature, radius, granularity, rate, count, or on/off switch explicitly named.
- **Revision signal**: a prefix item marked as superseded by a later observed item, or wording like "replace", "instead of", "reparameterize".

Almost every observed next decision acts on an object already present in the prefix or on the immediately adjacent missing slot of the same pipeline. Do **not** predict a new unrelated object, a new task domain, a theory/analysis section, or a baseline comparison with outside work unless the prefix already names it.

## Step 2 — Apply gap rules in priority order

1. **Data present, measurement absent → predict `experiment`: define how outcomes are scored.** This is the most reliable transition. Typical form: an evaluation protocol, error/quality metric, model-judge scoring, conditional-independence or correlation diagnostic, or a pass/fail criterion applied to the existing data. Also fires when the first decision is a mechanism plus a chosen corpus but no metric.

2. **Data and measurement both present, mechanism absent → predict `method`: introduce the project's core mechanism** (objective, estimator, training procedure) or the data-construction/labeling pipeline that feeds it. Prefixes that open with an evaluation suite, pilot comparison, or baseline comparison already containing metrics move to method next.

3. **Mechanism present, no data and no measurement → choose between two continuations.** Predict another `method` decision when the prefix mechanism looks provisional or partial: a coarse prompt-based heuristic, a degenerate parameter trick, one stage of an obviously multi-stage system, or a component with adjacent uncovered stages (target construction, labeling, loss term, efficiency, resolution/granularity choice). The usual operation is *replace with a finer-grained variant of the same stage* or *add the next stage*. Predict an `experiment` setting up data/metrics instead when the mechanism reads as complete and self-contained (a full detector, formulation, or loss with nothing obviously missing).

4. **Mechanism just introduced with one salient continuous knob or discrete granularity, and no evaluation exists yet → predict `ablation`: sweep that knob.** Prefer this over rule 3 when the prefix foregrounds a parameter range, a smoothness/scale factor, or a resolution choice as the mechanism's defining degree of freedom.

5. **Mechanism plus measurement present, no ablation yet → predict `ablation` over the newest named tunable or the most recently introduced component.** Early ablations are always sweeps of, or on/off toggles of, something already written in the prefix (a threshold, a scale, a prompt variant, a sample population, a structural alternative), never novel machinery.

6. **Prefix ends on a narrow synthetic/toy experiment or ablation → predict `experiment` that widens scope**: more conditions, a broader or differently spaced parameter range, more datasets from the same family, or the same measurement transferred from the synthetic setting to a real pretrained/real-data setting.

7. **Revision signal present.** If a method stage has already been superseded and no evaluation of it exists, predict yet another `method` revision of that same stage. If an ablation or experiment already targeted the pre-revision version, predict re-running or extending that same ablation/experiment under the revised version.

8. **All-method prefixes of length two or three with no data or metric anywhere** continue in `method`, advancing along the pipeline (construction → granularity/scope → objective → estimation/efficiency/validation). Switch to `experiment` only when every core stage the prefix implies is already specified; then predict the dataset/condition configuration.

9. **All-experiment prefixes** continue in `experiment`, alternating between data/condition generation and diagnostics/metrics on those conditions, until both are covered.

## Step 3 — Fallback for ambiguous prefixes

If two rules conflict or none clearly fires, predict the **same category as the most recent decision, acting on the same object, as an incremental extension or replacement of it**. This is the modal behavior at all three prefix lengths. Prefer `method` over `experiment` when nothing in the prefix has yet been measured; prefer `experiment` when data exists but nothing has been scored.

## What cannot be inferred this early

State predictions at the object-and-operation level. Do **not** commit to:

- numeric sweep values, thresholds, learning rates, sample counts, seeds, or split boundaries;
- the identity of specific external models, tokenizers, or judge models;
- the full list of datasets in a later evaluation suite (name the family implied by the prefix instead);
- whether the next decision will be retained, superseded, or reverted;
- long-run architecture choices, theory results, or final comparisons.

When rule 5 or 6 fires, name the swept knob taken verbatim from the prefix vocabulary and say the range is underdetermined. When rule 1 fires, name the measurement type suggested by the prefix's object (e.g., ranking/threshold quality for detection-style objects, reconstruction/forecast error for generative or emulation objects, correctness scoring for question-answering objects) rather than inventing a specific metric formula.

## Output form

One sentence naming the category, then one or two sentences naming the object plus operation, then one clause stating the chief alternative you rejected and which inventory gap decided it. Keep specifications generic wherever the prefix does not pin them down.
