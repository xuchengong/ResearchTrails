You are predicting the next research decision in a project based on a chronological record of historical research decisions.

A research decision changes the method, experiment, or ablation. Repository maintenance, routine implementation, reporting and logging, and arbitrary configuration changes are not research decisions unless they instantiate a scientifically meaningful choice.

Each historical research decision has an outcome/state label, including "retained by cutoff", "abandoned by cutoff", "superseded by observed Txxx", and "active at cutoff" for decisions whose evidence overlaps the target decision's starting time. Outcomes not established by the cutoff are withheld.

Predict the single most likely next research decision, not a summary or a list of possibilities. State what component is acted on and what changes about it. Name a dataset, model, metric, or numeric setting only where the observed record supports it; do not invent settings in order to sound specific. A short prefix may leave the answer uncertain; still select one most likely decision.

Return exactly one JSON object with two fields and no Markdown:
{"category": "method|experiment|ablation", "decision": "the next research decision"}

Do not output future outcomes, a trajectory insight, evidence, or an explanation of your reasoning.
