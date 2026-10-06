You are predicting the next research decision in a project based on a chronological record of historical research decisions.

A research decision changes the method, experiment, or ablation. Repository maintenance, routine implementation, reporting and logging, and arbitrary configuration changes are not research decisions unless they instantiate a scientifically meaningful choice.

Each historical research decision has an outcome/state label, including "retained by cutoff", "abandoned by cutoff", "superseded by observed Txxx", and "active at cutoff" for decisions whose evidence overlaps the target decision's starting time.

Based on the project's current research question, hypothesis, method, and evaluation logic, predict the single most likely next research decision. Decide in this order: first what kind of decision it is, then which component of the project it acts on, then what operation is applied to that component, and only then state the decision as a sentence. It must be one decision, neither too coarse nor too concrete.

Name a dataset, model, metric, or numeric setting only where the observed record supports it; do not invent settings in order to sound specific.

Return exactly one JSON object with this structure and no Markdown:

{
  "category":  "method|experiment|ablation",
  "object":    "the component acted on: a method component, metric, evaluation protocol, dataset or data-construction step, training stage, or baseline",
  "operation": "introduce|replace|extend|compare|remove|calibrate|validate|restrict",
  "decision":  "one sentence stating the decision"
}
