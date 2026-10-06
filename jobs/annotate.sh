# Build annotated research trajectories for one venue and year (paper Section 3).
# Run from the repository root:  VENUE=NeurIPS YEAR=2025 bash jobs/annotate.sh
# Each stage is resumable; rerunning skips completed work. 
# Pass --limit/--offset --indices to the individual scripts to process a subset.
set -euo pipefail

source .venv/bin/activate
set -a
source .env
set +a

VENUE=${VENUE:-NeurIPS}
YEAR=${YEAR:-2025}
venue=$(echo "$VENUE" | tr '[:upper:]' '[:lower:]')

# 1. Discover paper/repository pairs and keep pairs with progressive histories.
python3 annotate/sample.py --venue "$VENUE" --year "$YEAR"
python3 annotate/filter.py --venue "$venue" --year "$YEAR"

# 2. Collect commit evidence from each progressive repository.
python3 annotate/extract.py --venue "$venue" --year "$YEAR"

# 3-4. Extract decisions from commits, verify citations, and annotate trajectories (writes annotations/<venue>_<year>/<idx>-<repo>/annotation.json).
python3 annotate/annotate_openrouter.py --venue "$venue" --year "$YEAR"
