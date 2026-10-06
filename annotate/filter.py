"""Fetch GitHub commit-history signals and apply heuristic filtering of paper-repo pairs.

Reads the paper/repo pairs in [venue]_[year]_buffer.json produced by `sample.py` and
applies these filtering rules to each `repo_url` in order, stopping at the first failure:

1. Liveness. An accessible repository is `live`. Otherwise the status records why
   (`dead_404`, `empty_repo`, `bad_url`, `http_<code>`) and the pair is bucketed `dead`.
2. Known exclusions. Multi-paper repositories whose folders or branches hold independent
   papers are excluded, because their repo-wide history mixes the target paper's commits
   with other papers'. Upstream libraries linked as a paper's code are excluded when their
   history before the cutoff does not contain the paper's implementation. A paper-specific
   repository is not excluded merely for containing or modifying framework code; those
   edits are part of the paper's evolution.
3. Total commit count. Fewer than MIN_COMMITS is a `code_dump`.
4. Paper date. The pair needs an original arXiv upload date to define its cutoff, which
   is the end of that month.
5. Fork history. A fork of an unrelated upstream project inherits commits it did not
   author. The parent's README decides which kind of fork it is, using the same rule
   `sample.py` uses to retain a repository: if the parent's README also contains the
   paper's full title, the parent is the authors' own earlier repository and the whole
   history is theirs. Otherwise every later measurement counts only commits at or after
   the fork's `created_at`, and a fork with fewer than MIN_PRE_PAPER_COMMITS commits of
   its own is bucketed `exclude_fork_upstream_history`. A non-fork repository always
   keeps its full history, since commits pushed from a pre-existing local clone are
   legitimately the authors' own.
6. Pre-paper commits. Fewer than MIN_PRE_PAPER_COMMITS in that window is a `code_dump`.
7. Span. The first and last commits in that window must be at least MIN_SPAN_DAYS apart.
   Post-paper activity never contributes to the span.
8. Research messages. Commit titles in that window (up to COMMIT_CAP) are written to
   `sampled_msgs` and matched against RESEARCH_RX, a stem-based research-process
   vocabulary. Fewer than MIN_RESEARCH_HITS matching titles is a `code_dump`;
   otherwise the pair is bucketed `progressive` and kept. This is a cost-control
   prefilter over commit *titles*; only annotate_openrouter.py reads diffs and decides what is
   actually a research decision.

The output is annotate/[venue]_[year]_filtered.csv, one row per sampled pair.
The CSV and its state sidecar are replaced atomically after every completed pair, so an
interrupted job resumes where it stopped.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    ARXIV_API,
    GITHUB_API,
    arxiv_abs_url,
    arxiv_original_date,
    atomic_write_csv,
    atomic_write_json,
    decode_readme_response,
    github_headers,
    normalize_arxiv_id,
    paper_month_cutoff,
    parse_repo,
    rate_limit_reset_wait,
    readme_names_paper,
    sleep_for_rate_limit,
    venue_slug,
)
from envload import load_env  # noqa: E402

load_env()

BASE = Path(__file__).resolve().parent
HDR = github_headers("sampling-study")

FILTER_VERSION = "sequential-prepaper-v4-fork-aware-stem-regex"
STATE_VERSION = 4
MIN_COMMITS = 5
MIN_SPAN_DAYS = 30
MIN_PRE_PAPER_COMMITS = 5
MIN_RESEARCH_HITS = 2

FIELDS = [
    "filter_version", "idx", "conference", "year", "title", "arxiv_id",
    "repo_url", "status", "filter_stage", "commit_count", "commits_read",
    "first_date", "last_date", "span_days", "paper_date", "paper_month",
    "commits_pre_paper", "commits_post_paper", "research_pre_paper",
    "is_fork", "fork_parent", "fork_scope", "own_history_since",
    "commits_own_history",
    "first_msg", "last_msg", "heuristic_bucket", "research_hits",
    "heuristic_evidence", "sampled_msgs",
]

# Research-process vocabulary for pre-paper commit titles. Two properties matter:
#
#   * There is no trailing \b, so each entry is a STEM: "train" matches training and
#     trainer, "ablat" matches ablation and ablate, "optimiz" matches optimizer and
#     optimization. An exact-word version of this list reached only 10.9% recall
#     against LLM-adjudicated commit verdicts, almost entirely because \btrain\b
#     cannot match "training".
#   * Every entry earns its place empirically. A term is included only if it fires on
#     the sampled corpus and, where it overlaps a generic word, only if the commits it
#     uniquely matches beat the 34.3% base rate of research-bearing commits. "data"
#     (25.5%) and "test" (30.4%) fall below that base rate and are deliberately absent:
#     they add matches while diluting the signal.
#
# This rule is a cost-control prefilter, not a scientific criterion. At best it reaches
# roughly 25% recall and 50% precision on individual commits, because commit titles
# often do not say what changed scientifically. Only annotate_openrouter.py, which reads diffs,
# decides whether a change is a research decision.
RESEARCH_RX = re.compile(
    r"\b("
    # method and training
    r"experiment|explor|analysis|analy[sz]e|method|algorithm|objective|loss|"
    r"train|test|pretrain|finetun|fine-tun|optimiz|converg|hyperparam|schedule|"
    r"learning rate|epoch|checkpoint|seed|regulari[sz]|normali[sz]|augment|"
    r"sample|sampler|solve|solver|posterior|process|"
    # evaluation and comparison
    r"eval|validat|metric|benchmark|baseline|ablat|sweep|probe|"
    r"accuracy|perplexity|psnr|ssim|bleu|auc|robustness|adversarial|"
    r"ood|zero-shot|few-shot|holdout|"
    # data and models
    r"data|model|reward|distill|quantiz|prune|calibrat|"
    r"ppo|dpo|sft|lora"
    r")",
    re.I,
)

# Only include repositories whose history mixes independent papers/projects, or upstream
# libraries whose pre-paper history lacks the paper's code (found by inspecting the
# extracted evidence). Repository size, framework code, or a vendored dependency is not
# enough.
KNOWN_EXCLUDES = {
    "shangtongzhang/deeprl": (
        "shared codebase and branches contain implementations for many papers; "
        "the default history is not specific to the linked paper"
    ),
    "microsoft/econml": (
        "shared research package implements methods from multiple papers; "
        "the repo-wide history is not specific to the linked paper"
    ),
    "google-research/google-research": (
        "independent papers/projects live in separate folders, so repo-wide commits "
        "mix unrelated research trajectories"
    ),
    "microsoft/lmops": (
        "independent papers/projects live in separate folders, so repo-wide commits "
        "mix unrelated research trajectories"
    ),
    "eleutherai/gpt-neox": (
        "upstream GPT-NeoX, not the paper's (KERPLE) development; the paper's own "
        "repository has no commits before the cutoff"
    ),
    "microsoft/lightgbm": (
        "general LightGBM development; the paper's gradient-quantization code was "
        "added after the cutoff"
    ),
}


def gh(path):
    """GET from GitHub with rate-limit awareness."""
    req = urllib.request.Request(GITHUB_API + path, headers=HDR)
    while True:
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as e:
            h = dict(e.headers)
            if e.code == 403 and h.get("X-RateLimit-Remaining") == "0":
                wait = rate_limit_reset_wait(h)
                print(f"  rate limited; sleeping {wait}s to reset", file=sys.stderr)
                time.sleep(wait)
                continue
            return e.code, h, e.read()


def throttle(headers):
    """Sleep only when the remaining budget is exhausted; otherwise burst."""
    rem = headers.get("X-RateLimit-Remaining")
    if rem is not None and int(rem) <= 1:
        wait = rate_limit_reset_wait(headers)
        print(f"  budget exhausted; sleeping {wait}s", file=sys.stderr)
        time.sleep(wait)


def fetch_arxiv_dates(arxiv_ids, retries=10):
    """Return each arXiv paper's original upload date from the Atom API."""
    requested = [normalize_arxiv_id(value) for value in arxiv_ids]
    requested = [value for value in requested if value]
    if not requested:
        return {}
    query = urllib.parse.urlencode({
        "id_list": ",".join(requested),
        "max_results": len(requested),
    })
    req = urllib.request.Request(
        f"{ARXIV_API}?{query}",
        headers={"User-Agent": os.environ.get("ARXIV_USER_AGENT", "researchtrails/1.0")},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                root = ET.fromstring(response.read())
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt + 1 < retries:
                wait = max(sleep_for_rate_limit(dict(exc.headers or {}), attempt), 3)
                print(
                    f"arXiv HTTP {exc.code}; retrying in {wait:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"arXiv request failed with HTTP {exc.code}: {body[:300]}"
                + (". Retry with --arxiv-date-source abs to read original upload dates "
                   "from abstract pages." if exc.code == 406 else "")
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt + 1 < retries:
                wait = max(min(2**attempt, 60), 3)
                print(
                    f"arXiv request failed; retrying in {wait:.0f}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"arXiv request failed: {exc}") from exc
        except ET.ParseError as exc:
            raise RuntimeError("arXiv returned malformed XML") from exc
    else:
        raise RuntimeError("arXiv request retries were exhausted")
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    found = {}
    for entry in root.findall("atom:entry", namespace):
        entry_url = entry.findtext("atom:id", default="", namespaces=namespace)
        entry_id = normalize_arxiv_id(entry_url.rsplit("/abs/", 1)[-1])
        published = entry.findtext("atom:published", default="", namespaces=namespace)
        if entry_id and published:
            found[entry_id] = published[:10]
    return {arxiv_id: found.get(arxiv_id, "") for arxiv_id in requested}


def fetch_arxiv_page_date(arxiv_id):
    """Read the original [v1] submission date, even when the page shows a revision."""
    url = arxiv_abs_url(arxiv_id)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": os.environ.get("ARXIV_USER_AGENT", "researchtrails/1.0")},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            page = response.read().decode("utf-8")
    except OSError as exc:
        raise RuntimeError(f"arXiv abstract request failed for {url}: {exc}") from exc
    try:
        return arxiv_original_date(page)
    except ValueError as exc:
        raise RuntimeError(f"{exc}: {url}") from exc


def commit_window(until=None, since=None):
    """GitHub commit-listing bounds. `since` excludes history inherited by a fork."""
    params = {}
    if until:
        params["until"] = until
    if since:
        params["since"] = since
    return params


def commit_title(commit):
    """Return the bounded first line of a GitHub commit message, if present."""
    lines = (commit["commit"].get("message") or "").splitlines()
    return lines[0][:160] if lines else ""


def fetch_commit_msgs(owner, name, cap, until=None, since=None):
    """Fetch newest-first commit dates and title lines within the given window."""
    msgs, page, per = [], 1, 100
    while len(msgs) < cap:
        params = {**commit_window(until, since), "per_page": per, "page": page}
        st, h, body = gh(
            f"/repos/{owner}/{name}/commits?{urllib.parse.urlencode(params)}"
        )
        if st != 200:
            raise RuntimeError(f"GitHub returned HTTP {st} while fetching commit messages")
        throttle(h)
        chunk = json.loads(body)
        if not chunk:
            break
        for c in chunk:
            msgs.append((c["commit"]["committer"]["date"], commit_title(c)))
        if len(chunk) < per:
            break
        page += 1
    return msgs[:cap]


def known_exclusion(owner, name):
    return KNOWN_EXCLUDES.get(f"{owner}/{name}".lower())


def last_page(link_header):
    if not link_header:
        return None
    m = re.search(r'[?&]page=(\d+)[^>]*>;\s*rel="last"', link_header)
    return int(m.group(1)) if m else None


def commit_count(owner, name, until=None, since=None):
    params = {**commit_window(until, since), "per_page": 1}
    st, headers, body = gh(
        f"/repos/{owner}/{name}/commits?{urllib.parse.urlencode(params)}"
    )
    if st != 200:
        return st, headers, None, []
    throttle(headers)
    commits = json.loads(body)
    count = (last_page(headers.get("Link", "")) or 1) if commits else 0
    return st, headers, count, commits


def commit_at_page(owner, name, page, until=None, since=None):
    params = {**commit_window(until, since), "per_page": 1, "page": page}
    return gh(f"/repos/{owner}/{name}/commits?{urllib.parse.urlencode(params)}")


def parent_names_same_paper(parent_full_name, title):
    """Whether a fork's parent is the authors' own earlier repository for this paper.

    `sample.py` retains a repository only when its README contains the paper's full
    title. Applying the same rule to the parent separates two very different forks: a
    fork of an unrelated upstream project, whose inherited commits are not this paper's
    research, from a fork of the authors' own repository, whose entire history is. The
    second case is common when a personal repository is forked into a lab organization
    long after the work was done, which makes the fork's `created_at` useless as a
    history boundary.

    Returns True, False, or None when the parent's README could not be read.
    """
    if not parent_full_name or not title:
        return False
    status, headers, body = gh(f"/repos/{parent_full_name}/readme")
    if status == 404:
        return False
    if status != 200:
        return None
    throttle(headers)
    return readme_names_paper(decode_readme_response(body), title)


def repo_metadata(owner, name):
    """Repository record used to tell a fork's own history from inherited history."""
    status, headers, body = gh(f"/repos/{owner}/{name}")
    if status != 200:
        return status, None
    throttle(headers)
    repo = json.loads(body)
    parent = repo.get("parent") or {}
    return status, {
        "is_fork": bool(repo.get("fork")),
        "fork_parent": parent.get("full_name") or "",
        # A fork's created_at is when it was forked, so commits older than it were
        # authored upstream and are not part of this paper's research history.
        "created_at": repo.get("created_at") or "",
    }


def days_between(first_date, last_date):
    first = datetime.fromisoformat(first_date.replace("Z", "+00:00"))
    last = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
    return (last - first).days


def research_messages(messages):
    """Pre-paper commit titles whose wording suggests research process."""
    return [message for message in messages if RESEARCH_RX.search(message)]


def base_result():
    return {field: "" for field in FIELDS}


def rejected(row, stage, evidence, bucket="code_dump"):
    row["filter_stage"] = stage
    row["heuristic_bucket"] = bucket
    row["heuristic_evidence"] = evidence
    return row


def measure(owner, name, paper_date, commit_cap, *, title):
    """Apply the sequential filters, stopping as soon as one fails."""
    row = base_result()
    status_code, _, count, latest = commit_count(owner, name)
    if status_code == 404:
        row["status"] = "dead_404"
        return rejected(row, "liveness", "GitHub returned HTTP 404", bucket="dead")
    if status_code == 409:
        row["status"] = "empty_repo"
        row["commit_count"] = 0
        return rejected(row, "liveness", "repository has no commits", bucket="dead")
    if status_code != 200:
        row["status"] = f"http_{status_code}"
        return rejected(
            row, "liveness", f"GitHub returned HTTP {status_code}", bucket="dead"
        )
    if not latest:
        row["status"] = "empty_repo"
        row["commit_count"] = 0
        return rejected(row, "liveness", "repository has no commits", bucket="dead")

    row["status"] = "live"
    row["commit_count"] = count
    exclusion = known_exclusion(owner, name)
    if exclusion:
        return rejected(
            row,
            "known_exclusion",
            exclusion,
            bucket="exclude_monorepo_or_library",
        )
    if count < MIN_COMMITS:
        return rejected(
            row, "commit_count", f"commit_count={count} is below {MIN_COMMITS}"
        )

    metadata_status, metadata = repo_metadata(owner, name)
    if metadata is None:
        return rejected(
            row,
            "fork_status",
            f"could not read repository metadata (HTTP {metadata_status})",
            bucket="measurement_error",
        )
    row["is_fork"] = metadata["is_fork"]
    row["fork_parent"] = metadata["fork_parent"]
    # Only a fork of an unrelated upstream project inherits history it did not author.
    # A non-fork repository, and a fork of the authors' own repository for this paper,
    # both keep their full history.
    since = None
    if metadata["is_fork"]:
        same_project = parent_names_same_paper(metadata["fork_parent"], title)
        if same_project is None:
            return rejected(
                row,
                "fork_status",
                f"could not read the README of parent {metadata['fork_parent']}",
                bucket="measurement_error",
            )
        row["fork_scope"] = "same_project" if same_project else "post_fork"
        if not same_project:
            since = metadata["created_at"]
    row["own_history_since"] = since or ""

    row["paper_date"] = paper_date or ""
    row["paper_month"] = paper_date[:7] if paper_date else ""
    if not paper_date:
        return rejected(
            row,
            "paper_date",
            "arXiv did not return an original upload date",
            bucket="paper_date_unavailable",
        )

    cutoff = paper_month_cutoff(paper_date)
    pre_status, _, pre_count, pre_latest = commit_count(owner, name, until=cutoff)
    if pre_status != 200:
        return rejected(
            row,
            "pre_paper_commits",
            f"could not count pre-paper commits (HTTP {pre_status})",
            bucket="measurement_error",
        )
    row["commits_pre_paper"] = pre_count
    row["commits_post_paper"] = max(count - pre_count, 0)

    if since:
        own_status, _, own_count, own_latest = commit_count(
            owner, name, until=cutoff, since=since
        )
        if own_status != 200:
            return rejected(
                row,
                "fork_history",
                f"could not count post-fork commits (HTTP {own_status})",
                bucket="measurement_error",
            )
        row["commits_own_history"] = own_count
        if own_count < MIN_PRE_PAPER_COMMITS:
            return rejected(
                row,
                "fork_history",
                f"fork of unrelated upstream {metadata['fork_parent']} has "
                f"{own_count} pre-paper commits of its own (created {since[:10]}); "
                f"the remaining {max(pre_count - own_count, 0)} were authored upstream",
                bucket="exclude_fork_upstream_history",
            )
        pre_count, pre_latest = own_count, own_latest
    else:
        row["commits_own_history"] = pre_count

    if pre_count < MIN_PRE_PAPER_COMMITS:
        return rejected(
            row,
            "pre_paper_commits",
            f"commits_pre_paper={pre_count} is below {MIN_PRE_PAPER_COMMITS}",
        )

    if not pre_latest:
        return rejected(
            row,
            "span_days",
            "pre-paper commit count was nonzero but no latest commit was returned",
            bucket="measurement_error",
        )
    latest_pre_commit = pre_latest[0]
    row["last_date"] = latest_pre_commit["commit"]["committer"]["date"]
    row["last_msg"] = commit_title(latest_pre_commit)

    oldest_status, oldest_headers, oldest_body = commit_at_page(
        owner, name, pre_count, until=cutoff, since=since
    )
    oldest_rows = json.loads(oldest_body) if oldest_body else []
    if oldest_status != 200 or not oldest_rows:
        return rejected(
            row,
            "span_days",
            f"could not fetch oldest pre-paper commit (HTTP {oldest_status})",
            bucket="measurement_error",
        )
    throttle(oldest_headers)
    oldest_pre_commit = oldest_rows[0]
    row["first_date"] = oldest_pre_commit["commit"]["committer"]["date"]
    row["first_msg"] = commit_title(oldest_pre_commit)
    row["span_days"] = days_between(row["first_date"], row["last_date"])
    if row["span_days"] < MIN_SPAN_DAYS:
        return rejected(
            row,
            "span_days",
            f"pre-paper span_days={row['span_days']} is below {MIN_SPAN_DAYS}",
        )

    pre_messages = fetch_commit_msgs(owner, name, commit_cap, until=cutoff, since=since)
    timeline = sorted(pre_messages, key=lambda item: item[0] or "")
    messages = [message for _, message in timeline]
    research = research_messages(messages)
    row["commits_read"] = len(timeline)
    row["research_hits"] = len(research)
    row["research_pre_paper"] = len(research)
    row["sampled_msgs"] = " | ".join(
        f"{(date or '')[:10]} {message}" for date, message in timeline
    )
    if len(research) < MIN_RESEARCH_HITS:
        return rejected(
            row,
            "research_hits",
            f"pre-paper research_hits={len(research)} is below {MIN_RESEARCH_HITS}",
        )
    return rejected(
        row,
        "complete",
        "; ".join(research[:3]),
        bucket="progressive",
    )


def resolve_path(value, default):
    """Resolve a CLI path, treating a bare filename as living beside this script."""
    path = Path(value) if value else Path(default)
    if path.is_absolute():
        return path
    if path.exists() or path.parent != Path("."):
        return path
    return BASE / path


def row_key(item):
    return f"{normalize_arxiv_id(item.get('arxiv_id'))}\t{item.get('repo_url', '')}"


def selected_rows(state_rows, sample):
    return [state_rows[row_key(item)] for item in sample if row_key(item) in state_rows]


BUCKET_COUNTERS = {
    "progressive": "progressive",
    "code_dump": "code_dump",
    "exclude_monorepo_or_library": "excluded",
    "exclude_fork_upstream_history": "forks",
    "dead": "dead",
}


def progress_counts(rows):
    counts = dict.fromkeys(
        ["progressive", "code_dump", "excluded", "forks", "dead", "other"], 0
    )
    for row in rows:
        counts[BUCKET_COUNTERS.get(row.get("heuristic_bucket"), "other")] += 1
    return counts


def print_progress(state_rows, sample):
    rows = selected_rows(state_rows, sample)
    completed, total = len(rows), len(sample)
    width = 30
    filled = width if total == 0 else int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    counts = progress_counts(rows)
    line = (
        f"[{bar}] {completed}/{total} | "
        f"progressive={counts['progressive']} "
        f"code_dump={counts['code_dump']} "
        f"excluded={counts['excluded']} "
        f"forks={counts['forks']} "
        f"dead={counts['dead']} "
        f"other={counts['other']}"
    )
    if sys.stdout.isatty():
        print(f"\r{line}", end="\n" if completed == total else "", flush=True)
    else:
        print(line, flush=True)


def apply_known_exclusions_to_cached_rows(rows):
    for row in rows.values():
        if row.get("status") != "live":
            continue
        parsed = parse_repo(row.get("repo_url"))
        exclusion = known_exclusion(*parsed) if parsed else None
        if exclusion:
            row["filter_stage"] = "known_exclusion"
            row["heuristic_bucket"] = "exclude_monorepo_or_library"
            row["heuristic_evidence"] = exclusion


def initial_state(buffer_path, offset, commit_cap):
    return {
        "state_version": STATE_VERSION,
        "filter_version": FILTER_VERSION,
        "buffer": str(buffer_path.resolve()),
        "offset": offset,
        "commit_cap": commit_cap,
        "arxiv_dates": {},
        "rows": {},
        "complete": False,
    }


def load_state(state_path, out_path, buffer_path, offset, commit_cap, restart):
    expected = initial_state(buffer_path, offset, commit_cap)
    if restart:
        return expected
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
    elif out_path.exists():
        with open(out_path, newline="") as f:
            old_rows = list(csv.DictReader(f))
        if old_rows and any(row.get("filter_version") != FILTER_VERSION for row in old_rows):
            sys.exit(
                f"{out_path} was produced by an older filter. Use --restart or a new --out path."
            )
        state = expected
        state["rows"] = {
            f"{normalize_arxiv_id(row.get('arxiv_id'))}\t{row.get('repo_url', '')}": row
            for row in old_rows
        }
        state["arxiv_dates"] = {
            normalize_arxiv_id(row.get("arxiv_id")): row.get("paper_date", "")
            for row in old_rows
            if normalize_arxiv_id(row.get("arxiv_id"))
        }
    else:
        return expected

    checks = {
        "state_version": STATE_VERSION,
        "filter_version": FILTER_VERSION,
        "buffer": str(buffer_path.resolve()),
        "offset": offset,
        "commit_cap": commit_cap,
    }
    mismatches = [key for key, value in checks.items() if state.get(key) != value]
    if mismatches:
        sys.exit(
            f"Cannot resume {state_path}: changed {', '.join(mismatches)}. "
            "Use --restart or a different --state/--out path."
        )
    state.setdefault("arxiv_dates", {})
    state.setdefault("rows", {})
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue", default=os.environ.get("VENUE", "neurips"))
    parser.add_argument("--year", type=int, default=int(os.environ.get("YEAR", "2025")))
    parser.add_argument("--buffer", default=os.environ.get("BUFFER"))
    parser.add_argument("--out", default=os.environ.get("OUT"))
    parser.add_argument("--state", default=os.environ.get("STATE"))
    parser.add_argument("--offset", type=int, default=int(os.environ.get("OFFSET", "0")))
    env_limit = int(os.environ["N"]) if os.environ.get("N") else None
    parser.add_argument("--limit", type=int, default=env_limit)
    parser.add_argument(
        "--commit-cap", type=int, default=int(os.environ.get("COMMIT_CAP", "500"))
    )
    parser.add_argument(
        "--arxiv-date-source", choices=["api", "abs"], default="api",
        help="fetch upload dates using the Atom API or abstract-page submission histories",
    )
    parser.add_argument(
        "--arxiv-batch-size",
        type=int,
        default=int(os.environ.get("ARXIV_BATCH_SIZE", "100")),
    )
    parser.add_argument(
        "--arxiv-sleep", type=float, default=float(os.environ.get("ARXIV_SLEEP", "3"))
    )
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = f"{venue_slug(args.venue)}_{args.year}"
    buffer_path = resolve_path(args.buffer, BASE / f"{prefix}_buffer.json")
    out_path = resolve_path(args.out, BASE / f"{prefix}_filtered.csv")
    state_path = resolve_path(args.state, out_path.with_suffix(".state.json"))
    with open(buffer_path) as f:
        all_items = json.load(f)
    end = None if args.limit is None else args.offset + args.limit
    sample = all_items[args.offset:end]
    state = load_state(
        state_path, out_path, buffer_path, args.offset, args.commit_cap, args.restart
    )
    apply_known_exclusions_to_cached_rows(state["rows"])
    state["complete"] = False
    atomic_write_json(state_path, state)
    atomic_write_csv(out_path, selected_rows(state["rows"], sample), FIELDS)

    missing_ids = []
    for item in sample:
        arxiv_id = normalize_arxiv_id(item.get("arxiv_id"))
        if arxiv_id and arxiv_id not in state["arxiv_dates"] and arxiv_id not in missing_ids:
            missing_ids.append(arxiv_id)
    date_batch_size = 1 if args.arxiv_date_source == "abs" else args.arxiv_batch_size
    for start in range(0, len(missing_ids), date_batch_size):
        batch = missing_ids[start:start + date_batch_size]
        if start:
            time.sleep(args.arxiv_sleep)
        if args.arxiv_date_source == "abs":
            state["arxiv_dates"][batch[0]] = fetch_arxiv_page_date(batch[0])
        else:
            state["arxiv_dates"].update(fetch_arxiv_dates(batch))
        atomic_write_json(state_path, state)
        print(
            f"arXiv dates: {min(start + len(batch), len(missing_ids))}/{len(missing_ids)}",
            flush=True,
        )

    completed_before = len(selected_rows(state["rows"], sample))
    if completed_before:
        print(f"resuming with {completed_before}/{len(sample)} rows complete", flush=True)
    print_progress(state["rows"], sample)
    for position, item in enumerate(sample, 1):
        key = row_key(item)
        if key in state["rows"]:
            continue
        idx = args.offset + position
        row = base_result()
        row.update({
            "filter_version": FILTER_VERSION,
            "idx": idx,
            "conference": item.get("conference", args.venue),
            "year": item.get("year", args.year),
            "title": item.get("title", ""),
            "arxiv_id": normalize_arxiv_id(item.get("arxiv_id")),
            "repo_url": item.get("repo_url", ""),
        })
        parsed_repo = parse_repo(item.get("repo_url"))
        if not parsed_repo:
            row["status"] = "bad_url"
            rejected(row, "liveness", "repository URL is not a GitHub repository", bucket="dead")
        else:
            paper_date = state["arxiv_dates"].get(row["arxiv_id"], "")
            measured = measure(
                *parsed_repo, paper_date, args.commit_cap, title=row["title"]
            )
            for field in FIELDS:
                if measured.get(field, "") != "":
                    row[field] = measured[field]

        state["rows"][key] = row
        atomic_write_json(state_path, state)
        atomic_write_csv(out_path, selected_rows(state["rows"], sample), FIELDS)
        print_progress(state["rows"], sample)

    state["complete"] = True
    atomic_write_json(state_path, state)
    atomic_write_csv(out_path, selected_rows(state["rows"], sample), FIELDS)
    print("DONE ->", out_path)
    print("STATE ->", state_path)

if __name__ == "__main__":
    main()
