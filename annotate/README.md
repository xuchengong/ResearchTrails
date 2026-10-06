# Annotating Research Decision Trajectories

The pipeline has four stages:

```
Semantic Scholar venue/year -> sample.py
-> repo pairs in <venue>_<year>_buffer.json -> filter.py 
-> venue>_<year>_filtered.csv -> extract.py 
-> ../annotations/<venue>_<year>/*/evidence.json -> annotate_openrouter.py 
-> ../annotations/<venue>_<year>/*/annotation.json
```

## Running it

Keys are read from `.env` at the repo root. `GITHUB_TOKEN` is needed since unauthenticated GitHub allows 60 requests/hour; `OPENROUTER_API_KEY` is needed for `annotate_openrouter.py`.

```bash
python3 annotate/sample.py     --venue NeurIPS --year 2025
python3 annotate/filter.py     --venue neurips --year 2025
python3 annotate/extract.py --venue neurips --year 2025 --limit 10 --offset 10
python3 annotate/annotate_openrouter.py   --venue neurips --year 2025 --limit 10 --offset 10
```

To regenerate selected pairs without touching the existing annotations:

```bash
python3 annotate/annotate_openrouter.py \
  --venue neurips --year 2025 \
  --indices 53 72 109 149 \
  --annotation-file annotation_new.json
```
The specific details of each stage is as follows.

## Stages

### 1. `sample.py` for candidate discovery

Retrieve the paper list for a venue/year from the Semantic Scholar bulk-search API, resolves each paper's arXiv id, and collects GitHub links from the arXiv abstract page and from the latest PDF's main text (everything before References/Appendix). A repository is retained only when its README contains the paper's title.

Writes `<venue>_<year>_buffer.json` (the list of retained pairs) and a resumable state `<venue>_<year>_buffer.state.json`.

`no_arxiv_id` means the cached Semantic Scholar record has no arXiv ID; no arXiv request was made for that paper. Undated proceedings records sort last, so these often cluster at the end. Download errors are recorded as `fetch_error`.

### 2. `filter.py` for deterministic filtering

Query GitHub for commit histories. Sequential filterings are as follows:

| Filtering rule | Meaning | Label upon failure |
|---|---|---|
| liveness | repo reachable and non-empty | `dead` |
| known exclusion | multi-paper monorepo or shared library | `exclude_monorepo_or_library` |
| commit count | at least 5 commits | `code_dump` |
| paper date | arXiv returns an original upload date | `paper_date_unavailable` |
| fork history | a fork of an unrelated upstream has at least 5 pre-paper commits of its own | `exclude_fork_upstream_history` |
| pre-paper commits | at least 5 commits at or before the arXiv upload month | `code_dump` |
| span | pre-paper commits span at least 30 days | `code_dump` |
| research messages | at least 2 commit titles match `RESEARCH_RX` | `code_dump` |

Surviving pairs are labelled as `progressive`. Writes and updates `<venue>_<year>_filtered.csv` and `.state.json` after every completed pair.

If the arXiv Atom API returns HTTP 406 but abstract pages are accessible, use:

```bash
python3 annotate/filter.py --venue NeurIPS --year 2023 --arxiv-date-source abs
```

This reads the paper upload date from its submission history. It fetches one page at a time with `--arxiv-sleep` seconds between requests (default 3), and checkpoints every date in the filter state. 

### 3. `extract.py` for commit collection

For each `progressive` pair, clones or refreshes a bare blob-limited mirror and walks the default branch **in commit-date order** up to the end of the arXiv upload month. Commit ordering uses `git rev-list --date-order`.

Writes `../annotations/<venue>_<year>/<idx>-<owner>-<name>/evidence.json`, updates `evidence_status` in the filtered CSV, and writes an aggregate file-type audit. Evidence is saved only after the entire repository finishes.

Before building a repository's evidence, extraction fetches arXiv version 1 of the paper and records its title, abstract and upload date in `evidence.json`; the commit cutoff is the end of that upload month. Reused evidence is not refetched. If the arXiv API returns HTTP 406, read the version-1 abstract page instead:

```bash
python3 annotate/extract.py --venue neurips --year 2023 --arxiv-metadata-source abs --retry-errors
```

This reads the same title, abstract and submission date from the abstract page. `--retry-errors` reopens rows that failed earlier. 

### 4. `annotate_openrouter.py` for commit accessment and annotation

Takes `evidence.json` as input and walks commits chronologically, greedily filling each request up to a model-aware input budget and an estimated delta-output budget. Model responses are **deltas**: only
changed or new decisions, newly added citations, changed threads, resolutions, and merges. The harness applies each delta to the complete local state, so unchanged decisions and citations are not repeated in model output. 

An LLM reads the commits in order, given the decisions so far, and labels each commit `irrelevant`, `support`, `revise`, `new`, `multiple` or `uncertain`. Uncertain candidates live only in `open_threads` until later evidence resolves them. Every thread removal must be recorded as `merged_into_decision`, `promoted_to_decision`, or `rejected`. 

Earlier decisions carry only semantic fields and IDs; the five most recent decisions and all unresolved threads carry their full stored citations (`RECENT_DECISIONS_WITH_FULL_EVIDENCE`).

Every claimed SHA, path, date, and exact diff excerpt is re-checked against `evidence.json`, and a citation that is not an exact substring of a stored patch is dropped. Decisions are ordered by their first cited commit date. If fewer than five survive, the row is marked `not_worth_annotating`; otherwise one final metadata call produces keywords and a trajectory insight from the validated decisions alone.

If the model omits a required commit assessment, returns malformed JSON, or produces an uncited thread, the response is discarded and the same commit range is retried with a smaller input budget.

## Output format

`annotation.json` is the packaged record:

```json
{
  "title": "...", "abstract": "...", "venue": "NeurIPS", "year": 2025,
  "source": {"arxiv": "https://arxiv.org/abs/...", "github": "https://github.com/..."},
  "keywords": ["..."],
  "decisions": [
    {
      "decision_id": "D003",
      "time_step_id": 0,
      "decision": "What was chosen, concretely.",
      "category": "method|experiment|ablation",
      "outcome": "retained|abandoned|superseded",
      "superseded_by": "D011",
      "first_date": "2024-09-06T...", "last_date": "2024-09-10T...",
      "evidence": [{"commit_sha": "...", "date": "...", "path": "...", "diff_excerpt": "..."}],
      "why_research_relevant": "..."
    }
  ],
  "decision_count": 12,
  "trajectory_insight": "..."
}
```

`outcome` records what later happened to a valid decision; A `superseded` decision names its successor in `superseded_by`, which refers to a `decision_id` in the same file — that link is what makes the ordered list a trajectory rather than a flat list. A link whose successor did not survive validation is dropped rather than published. `time_step_id` is the zero-based position after timezone-aware ordering by `first_date`, then `last_date`, then `decision_id` as a deterministic tie-breaker.



### Cost

`annotate_openrouter.py` is the only paid stage, and cost scales with commit count. For NeurIPS 2025: $1.56–$10.46 per annotated project (45–373 commits). It is recommended to check `commit_count` in the filtered CSV before annotation; If the commit count is large, one can pass an LLM call to re-examine the research-relevance of each commit and remove artifacts before annotation.
