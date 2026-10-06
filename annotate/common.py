"""Helpers shared by the sampling, filtering, extraction, and annotation stages.

Every function here is used by at least two stages. Stage-specific policy stays with
the stage that owns it: request retry types, per-stage CLI defaults, evidence rules,
and the LLM request budget all live in their own module.

Two slug functions exist because the pipeline slugs two different things. Generated
filenames use ``venue_slug`` and must agree across every stage, or a later stage stops
finding an earlier stage's output. Project directory names use ``path_slug``, which
keeps ``.``, ``-``, and ``_`` so an owner or repository name survives intact.

Two arXiv-id functions exist for the same reason. ``extract_arxiv_id`` recovers an id
from arbitrary paper metadata and returns "" when the value is not an arXiv id at all.
``normalize_arxiv_id`` canonicalizes an id that is already known to be one and never
discards it.
"""

import base64
import calendar
import csv
import html
import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

GITHUB_API = "https://api.github.com"
ARXIV_API = "https://export.arxiv.org/api/query"


def venue_slug(value):
    """Slug for generated filenames. Must match across all stages."""
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_") or "venue"


def path_slug(value):
    """Slug for a project directory component. Keeps dots, dashes, underscores."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-.")
    return value or "unknown"


def normalize_arxiv_id(value):
    """Canonicalize an arXiv id that is already known to be one."""
    value = str(value or "").strip()
    value = re.sub(r"^arxiv:\s*", "", value, flags=re.I)
    value = re.sub(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", value, flags=re.I)
    value = re.sub(r"\.pdf(?:\?.*)?$", "", value, flags=re.I)
    return re.sub(r"v\d+$", "", value)


def extract_arxiv_id(value):
    """Recover an arXiv id from arbitrary metadata, or "" when there is none."""
    if not value:
        return ""
    value = str(value).strip()
    value = re.sub(r"^ARXIV:", "", value, flags=re.I)
    match = re.search(r"10\.48550/arxiv\.([^?#\s]+)", value, flags=re.I)
    if match:
        value = match.group(1)
    match = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^?#\s]+)", value, flags=re.I)
    if match:
        value = match.group(1)
    value = value.removesuffix(".pdf").strip("/")
    match = re.match(r"([A-Za-z.-]+/\d{7})(?:v\d+)?$", value)
    if match:
        return match.group(1)
    match = re.match(r"(\d{4}\.\d{4,5})(?:v\d+)?$", value)
    if match:
        return match.group(1)
    return ""


def arxiv_abs_url(arxiv_id):
    return f"https://arxiv.org/abs/{normalize_arxiv_id(arxiv_id)}"


def arxiv_original_date(page):
    """Read the original [v1] date from an arXiv abstract page's submission history."""
    history = re.search(
        r'<div\b[^>]*class=["\']submission-history["\'][^>]*>(.*?)</div>',
        page, flags=re.S | re.I,
    )
    text = html.unescape(re.sub(r"<[^>]*>", "", history.group(1))) if history else ""
    text = " ".join(text.split())
    original = re.search(
        r"\[v1\]\s+([A-Za-z]{3}, \d{1,2} [A-Za-z]{3} \d{4} "
        r"\d{2}:\d{2}:\d{2} (?:UTC|GMT))\b", text,
    )
    if not original:
        raise ValueError("arXiv page has no original [v1] submission date")
    return parsedate_to_datetime(original.group(1)).date().isoformat()


def parse_repo(url):
    """Split a GitHub URL into (owner, name), or None when it is not one."""
    match = re.search(r"github\.com/([^/]+)/([^/#?]+)", str(url or ""))
    if not match:
        return None
    return match.group(1), re.sub(r"\.git$", "", match.group(2))


def paper_month_cutoff(paper_date):
    """Last instant of the month a paper was first uploaded to arXiv."""
    parsed = datetime.strptime(paper_date, "%Y-%m-%d")
    last_day = calendar.monthrange(parsed.year, parsed.month)[1]
    return f"{parsed.year:04d}-{parsed.month:02d}-{last_day:02d}T23:59:59Z"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value, default):
    """Interpret a CLI path: absolute as given, otherwise relative to the cwd."""
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def atomic_write_csv(path, rows, fieldnames):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", restval=""
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def github_headers(user_agent, accept="application/vnd.github+json"):
    headers = {
        "Accept": accept,
        "User-Agent": user_agent,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return headers


def decode_readme_response(body):
    """Decode a GitHub /readme response: raw bytes, or a base64 JSON object."""
    if not body.lstrip().startswith(b"{"):
        return body.decode("utf-8", "replace")
    try:
        payload = json.loads(body)
        content = payload.get("content") or ""
        if payload.get("encoding") == "base64" and content:
            return base64.b64decode(content).decode("utf-8", "replace")
    except (ValueError, TypeError):
        pass
    return body.decode("utf-8", "replace")


def collapse_whitespace(value):
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip().casefold()


def readme_names_paper(readme, title):
    """The study's retention rule: a README contains the paper's full title."""
    normalized_title = collapse_whitespace(title)
    return bool(normalized_title) and normalized_title in collapse_whitespace(readme)


def rate_limit_reset_wait(headers, minimum=5):
    """Seconds to wait for the GitHub rate-limit window to reset."""
    reset = int(headers.get("X-RateLimit-Reset", "0"))
    return max(reset - int(time.time()) + 3, minimum)


def sleep_for_rate_limit(headers, attempt):
    """Seconds to wait before retrying, honouring Retry-After when present."""
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 1)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 1)
            except (TypeError, ValueError, OverflowError):
                pass
    reset = headers.get("X-RateLimit-Reset")
    remaining = headers.get("X-RateLimit-Remaining")
    if reset and remaining == "0":
        return max(int(reset) - int(time.time()) + 3, 5)
    return min(2 ** attempt, 60)


def retryable_http_error(status, headers, body, *, allow_403_retry=True):
    """Whether an HTTP failure is transient. 403 is retryable only when throttled."""
    if status in {429, 500, 502, 503, 504}:
        return True
    if status != 403 or not allow_403_retry:
        return False
    lowered = body.lower()
    return (
        headers.get("X-RateLimit-Remaining") == "0"
        or "rate limit exceeded" in lowered
        or "secondary rate limit" in lowered
    )
