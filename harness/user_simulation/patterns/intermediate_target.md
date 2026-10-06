# Introduce an intermediate target when the full change is too difficult

Use this pattern when direct optimization toward a demanding goal stalls and there is a credible, easier step that teaches capabilities needed for that same goal.

State transition:
Direct attempt at the final goal stalls → identify a missing prerequisite or an excessively large change → choose an attainable intermediate target → train or construct an intermediate solution → carry that solution into the harder stage → evaluate against the original final goal.

The intermediate target should contribute to the final task, rather than merely being easier. It may be a simpler task distribution, a less stringent objective, a moderately changed model, or a preparatory training objective. Specify what is transferred between stages and what observation would justify moving forward. Keep the final evaluation fixed, account for the extra training cost, and compare with direct optimization under a comparable total budget. Treat the value of the intermediate target as a hypothesis to test, not as a guaranteed improvement.
