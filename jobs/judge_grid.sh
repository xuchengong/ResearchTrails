#!/usr/bin/env bash
# Opus 5 judge grid (paper Appendix B, the Opus 5 rows of the judge tables): Opus 5 re-judges
# the reported predictions. Each request is the saved Sol request that produced a reported
# score with the rubric and judge replaced, so run this after the other harness jobs and the
# RL evaluation runs of Section 3; the saved Sol judgments fill the Sol column.
source jobs/common.sh

# main: the harness repeats, the RL runs, and the calibration controls.
# judge-variance: the 100 fixed GPT-5.6 Luna predictions, judged three times.
for SUITE in main judge-variance; do
    python harness/judge_grid.py plan --suite "$SUITE"
    python harness/judge_grid.py prepare --suite "$SUITE"
    python harness/judge_grid.py run --suite "$SUITE" --concurrency "$CONCURRENCY" --confirm-submit
    python harness/judge_grid.py summarize --suite "$SUITE"
done
