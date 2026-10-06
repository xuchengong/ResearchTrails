Distill the supplied research decision trajectories into an Agent Skill for predicting the one most likely next scientific decision in an ongoing project.

Derive the skill's guidance from the supplied trajectories themselves. Infer which recurring patterns are predictive and portable across projects. Do not import a predefined model of how research should progress.

Return only a complete raw `SKILL.md`, beginning with YAML frontmatter and without a Markdown code fence. Use the required skill name given with the records. The body should contain 400-1500 words.

Do not copy project names, model names, datasets, numerical hyperparameters, decision identifiers, or domain-specific solutions into the skill. Avoid generic research advice that the trajectories do not include, repeated rules, taxonomies added only for organization, and instructions that do not materially change next-decision prediction behavior.

The research decision records follow after this instruction.
