# Audit the premise against the cheapest alternative

Use this pattern when work concentrates on improving a method that rests on an assumption nobody has tested in the user's setting: for example, that a diagnostic identifies what to intervene on, or that the method's selection step is what produces the observed gain. A gain can come from something cheaper that the assumption ignores.

State transition:
Method rests on an untested premise → state what the premise predicts → identify the cheapest alternative that could produce the same gain → test the premise's prediction against that alternative under matched conditions → keep, narrow, or drop the premise → only then decide whether refining the method is worthwhile.

1. State the premise explicitly and the observable consequence it implies. Separate what current results show from what the premise assumes.
2. Identify the cheapest competing lever a skeptical reader would try first: a trivial baseline or covariate, light tuning of the existing baseline, or a stronger standard training setup.
3. Design a comparison in which the premise and the cheap alternative predict different outcomes. Match budgets, data, and tuning effort across arms.
4. Decide beforehand how each outcome changes the plan. If the premise survives, refine the method. If the cheap lever matches it, the research question changes. If the premise holds only in some regimes, narrow the claim to them.
5. Invest in refining the method only after its premise has survived; refinements of a method whose premise fails do not transfer.

Apply this pattern only where warranted. Do not assume the premise is false, and do not invent results for either arm.
