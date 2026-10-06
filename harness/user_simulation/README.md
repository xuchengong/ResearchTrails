# User Simulation

`run.py` produces the paper's user-simulation figure. It asks a model the same research question twice: once plainly and once with a research pattern appended to its instructions. The comparison is qualitative.

The figure shows GPT-5.6 Sol's answers to two queries:

| Query | Pattern | 
|---|---|
| `queries/bug_repair.md`: reinforcement learning for a bug-repair agent rarely succeeds | `patterns/intermediate_target.md` |
| `queries/answer_judge.md`: an LLM judge picks the best of eight sampled answers | `patterns/premise_audit.md` | 

The patterns describe general research moves. `intermediate_target` generalizes a move found in trajectories 573 (Compiler-R1, NeurIPS 2025), 923 (MTS3, 2023), 121 (CeSoR, 2022), and 401 (LambdaBeam, 2023); `premise_audit` one found in NeurIPS 2023 trajectories 3 (belief-localization), 173 (tabzilla), 16 (realistic-al), and 366 (Trade-Off-MOL).

Each run makes two calls to GPT-5.6 Sol (`openai/gpt-5.6-sol`), without and with the pattern. Both conditions share the same base instructions. Calls use high reasoning effort and an 8K-token output budget.

## Run

From the repository root, with `OPENROUTER_API_KEY` loaded from `.env`:

```bash
set -a; source .env; set +a

python3 harness/user_simulation/run.py bug_repair     # -> harness/user_simulation/runs/bug_repair/
python3 harness/user_simulation/run.py answer_judge   # -> harness/user_simulation/runs/answer_judge/
```

These make paid API calls. Add `--dry-run` to write and inspect the prepared requests without calling the API.

Each run directory contains:

- `COMPARISON.md`: the query and both answers with condition labels.
- `requests.json`: the exact prepared request bodies.
- `attempts.jsonl`: start and completion records, raw API responses, and errors.
- `usage.json`: returned model names, response IDs, and usage for successful calls.

Rerunning the same command reuses completed responses. A changed query, pattern, or generation setting requires a new `--run-dir`. Failed or interrupted attempts require `--retry-errors` after inspecting the ledger.

Models, repetitions, reasoning effort, output budget, concurrency, and timeout are configurable; see `--help`.
