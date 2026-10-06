Distill the supplied prefix-to-next-decision training examples into an Agent Skill for predicting the one most likely next scientific decision near the natural beginning of an ongoing project.

The consumer sees the project's complete observable history, but that history contains only its first one, two, or three decisions because the project is genuinely at an early stage. No older decisions are missing. Guidance for a mature project with hidden earlier history is not applicable here.

Every training example contains only a cutoff-valid observable prefix followed by the actual next annotated decision. Analyze the first-one, first-two, and first-three conditions separately before deriving rules that transfer across them. Use only information present in the observable prefix: the scientific object, operation, category, apparent stage, and protocol vocabulary. Do not make a rule depend on the actual next decision or on later examples from the same project being visible at prediction time.

Prefer direct, executable rules supported by recurring transitions across multiple projects. Emphasize which object is likely to be acted on and which operation is likely to follow. Treat concrete specifications as underdetermined unless the short prefix strongly constrains them. State what cannot be inferred this early and give a conservative fallback for ambiguous cases.

Derive the guidance from the supplied examples rather than importing a predefined account of how research ought to progress. Do not reconstruct retained cores, supersession chains, long-term refinement debts, or final project outcomes: those require history that an early-stage predictor does not possess.

Return only a complete raw `SKILL.md`, beginning with YAML frontmatter and without a Markdown code fence. Use the required skill name given with the records. The body must contain 400-1500 words.

Do not copy project names, model names, datasets, numerical hyperparameters, decision identifiers, or domain-specific solutions into the skill. Avoid generic research advice unsupported by the examples, repeated rules, decorative taxonomies, and instructions that do not materially change next-decision prediction behavior.

The cutoff-local prefix-to-next-decision examples follow after this instruction.
