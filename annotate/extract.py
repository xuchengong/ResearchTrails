"""Prepare repository evidence, which will be input to LLM for annotation.

For each selected ``progressive`` paper/repository pair, this resumable stage:

1. Loads the title, abstract and upload date of arXiv version 1 and computes the end
   of that upload month.
2. Clones or refreshes the repository and walks relevant commits chronologically up
   to that cutoff, optionally enriching the history with merged pull requests.
3. Writes one auditable manifest record for every changed file. Binary files,
   generated artifacts, logs, and generated-folder contents do not include content.
4. Applies ``read_at_creation`` to source, configuration, and notebook files. On the
   first add event, it stores the file's complete text up to a type-specific cap.
   Python has a larger default cap than shell scripts. Every later event stores only
   the bounded Git diff, so an unchanged file is never sent repeatedly.
5. Stores ``evidence.json``, updates ``evidence_status`` in the filtered CSV, and
   writes an aggregate file-type audit.

No before/after file bodies are collected. A long creation snapshot is explicitly
truncated while recording its original length; subsequent diffs have their own cap.
An existing project evidence file is reused unless ``--force`` is passed. Progress
within an unfinished repository is not checkpointed; restarting reuses its Git
cache and extracts that repository's evidence again from its first commit.
``annotate_openrouter.py`` is the separate consumer of this output.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    ARXIV_API,
    GITHUB_API,
    arxiv_abs_url,
    arxiv_original_date,
    atomic_write_csv,
    atomic_write_json,
    github_headers,
    normalize_arxiv_id,
    paper_month_cutoff,
    parse_repo,
    path_slug,
    resolve_path,
    retryable_http_error,
    sleep_for_rate_limit,
    utc_now,
    venue_slug,
)
from envload import load_env  # noqa: E402

load_env()

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parent
TRUNCATION_MARKER = "characters omitted by evidence policy"
DEFAULT_GIT_BLOB_LIMIT = int(
    os.environ.get("EXTRACTION_GIT_BLOB_LIMIT", str(2 * 1024 * 1024))
)
DEFAULT_CREATION_LIMITS = {
    "python": 50000,
    "shell": 20000,
    "source": 50000,
    "configuration": 20000,
    "notebook": 50000,
}
CSV_ADDITIONS = ["evidence_status", "evidence_error", "abstract", "paper_date"]

GENERATED_FOLDER_MARKERS = ("output", "result", "checkpoint", "cache")

BINARY_EXTENSIONS = {
    ".7z",
    ".a",
    ".arrow",
    ".avi",
    ".bin",
    ".bmp",
    ".bz2",
    ".ckpt",
    ".class",
    ".dll",
    ".dylib",
    ".feather",
    ".gif",
    ".gz",
    ".h5",
    ".hdf5",
    ".ico",
    ".jar",
    ".joblib",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".o",
    ".onnx",
    ".otf",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".so",
    ".tar",
    ".tflite",
    ".ttf",
    ".wav",
    ".webm",
    ".webp",
    ".xz",
    ".zip",
}

SOURCE_EXTENSIONS = {
    ".bash",
    ".c",
    ".cc",
    ".clj",
    ".cljs",
    ".coffee",
    ".cpp",
    ".cs",
    ".cu",
    ".cuh",
    ".dart",
    ".ex",
    ".exs",
    ".f",
    ".f90",
    ".go",
    ".groovy",
    ".h",
    ".hpp",
    ".hs",
    ".java",
    ".jl",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".lua",
    ".m",
    ".mm",
    ".php",
    ".pl",
    ".pm",
    ".ps1",
    ".py",
    ".pyx",
    ".r",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".swift",
    ".tcl",
    ".ts",
    ".tsx",
    ".vue",
}
PYTHON_EXTENSIONS = {".py", ".pyx"}
SHELL_EXTENSIONS = {".bash", ".sh"}

CONFIG_EXTENSIONS = {
    ".cfg",
    ".conf",
    ".config",
    ".env",
    ".ini",
    ".properties",
    ".toml",
    ".xml",
    ".yaml",
    ".yml",
}

DOCUMENT_EXTENSIONS = {".md", ".markdown", ".rst", ".tex"}
AMBIGUOUS_TEXT_EXTENSIONS = {
    "",
    ".csv",
    ".json",
    ".jsonl",
    ".list",
    ".tsv",
    ".txt",
}
MANIFEST_ONLY_EXTENSIONS = {".log"}

SOURCE_FILENAMES = {
    "cmakelists.txt",
    "dockerfile",
    "gemfile",
    "justfile",
    "makefile",
    "meson.build",
    "rakefile",
    "snakefile",
}
CONFIG_FILENAME_PATTERNS = (
    re.compile(r"^requirements(?:[-_.].*)?\.txt$"),
    re.compile(r"^environment(?:[-_.].*)?\.ya?ml$"),
    re.compile(r"^docker-compose(?:[-_.].*)?\.ya?ml$"),
)


class FilePolicy(NamedTuple):
    category: str
    mode: str
    read_at_creation: bool
    creation_limit_kind: str
    reason: str


class NotebookFormatError(ValueError):
    """Raised when a notebook cannot be reduced to auditable source cells."""


class ExtractionError(RuntimeError):
    """A recoverable per-project evidence-extraction failure."""


class HttpStatusError(ExtractionError):
    def __init__(self, status, url, body="", headers=None):
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body
        self.headers = dict(headers or {})


def extension(path):
    return PurePosixPath(str(path or "").replace("\\", "/")).suffix.lower()


def generated_folder(path):
    parts = PurePosixPath(str(path or "").replace("\\", "/")).parts
    for component in parts[:-1]:
        lowered = component.lower()
        if any(marker in lowered for marker in GENERATED_FOLDER_MARKERS):
            return component
    return ""


def classify_file(path, *, binary=False):
    path = str(path or "")
    filename = PurePosixPath(path.replace("\\", "/")).name.lower()
    suffix = extension(path)
    generated = generated_folder(path)
    if generated:
        return FilePolicy(
            "generated_artifact",
            "manifest_only",
            False,
            "",
            f"content omitted because directory {generated!r} matches a generated-folder marker",
        )
    if binary or suffix in BINARY_EXTENSIONS:
        return FilePolicy(
            "binary_or_model_artifact",
            "manifest_only",
            False,
            "",
            "content omitted because the file is binary or uses a binary/model/data format",
        )
    if suffix in MANIFEST_ONLY_EXTENSIONS:
        return FilePolicy(
            "generated_log",
            "manifest_only",
            False,
            "",
            "content omitted because log files are generated execution artifacts",
        )
    if suffix == ".pgn":
        return FilePolicy(
            "generated_artifact",
            "manifest_only",
            False,
            "",
            "content omitted because PGN files contain chess-game records",
        )
    if suffix == ".ipynb":
        return FilePolicy(
            "notebook_source",
            "notebook_source",
            True,
            "notebook",
            "notebook source is read at creation; later changes contain source-only diffs",
        )
    if suffix in SOURCE_EXTENSIONS or filename in SOURCE_FILENAMES:
        if suffix in PYTHON_EXTENSIONS:
            creation_kind = "python"
        elif suffix in SHELL_EXTENSIONS:
            creation_kind = "shell"
        else:
            creation_kind = "source"
        return FilePolicy(
            "source_code",
            "source_patch",
            True,
            creation_kind,
            "source is read at creation; later changes contain only bounded diffs",
        )
    if suffix in CONFIG_EXTENSIONS or any(
        pattern.match(filename) for pattern in CONFIG_FILENAME_PATTERNS
    ):
        return FilePolicy(
            "configuration",
            "source_patch",
            True,
            "configuration",
            "configuration is read at creation; later changes contain only bounded diffs",
        )
    if suffix in DOCUMENT_EXTENSIONS:
        return FilePolicy(
            "documentation",
            "text_preview",
            False,
            "",
            "small documentation patch preview retained; documentation alone is weak evidence",
        )
    if suffix in AMBIGUOUS_TEXT_EXTENSIONS:
        return FilePolicy(
            "ambiguous_text",
            "text_preview",
            False,
            "",
            "ambiguous text/data file receives only a bounded patch preview",
        )
    return FilePolicy(
        "unknown_text",
        "text_preview",
        False,
        "",
        "unrecognized non-binary file receives only a bounded patch preview",
    )


def normalize_notebook(raw):
    if raw in {None, b"", ""}:
        return ""
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        notebook = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotebookFormatError(f"invalid notebook JSON: {exc}") from exc
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        raise NotebookFormatError("notebook JSON has no cells list")
    rendered = []
    for index, cell in enumerate(cells, 1):
        if not isinstance(cell, dict):
            continue
        cell_type = str(cell.get("cell_type") or "unknown")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(str(part) for part in source)
        elif not isinstance(source, str):
            source = str(source)
        rendered.append(f"### cell {index} [{cell_type}]\n")
        rendered.append(source)
        if source and not source.endswith("\n"):
            rendered.append("\n")
    return "".join(rendered)


def notebook_source_diff(before_raw, after_raw, old_path, path):
    before = normalize_notebook(before_raw)
    after = normalize_notebook(after_raw)
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{old_path or path}",
            tofile=f"b/{path}",
            n=3,
        )
    )


def limit_text(value, max_chars):
    value = value or ""
    if len(value) <= max_chars:
        return value, False, 0
    marker_template = "\n[... {omitted} characters omitted by evidence policy ...]\n"
    marker = marker_template.format(omitted=0)
    if max_chars <= len(marker):
        return value[:max_chars], True, len(value) - max_chars
    keep = max(max_chars - len(marker), 0)
    left = keep // 2
    right = keep - left
    omitted = max(len(value) - left - right, 0)
    marker = marker_template.format(omitted=omitted)
    keep = max(max_chars - len(marker), 0)
    left = keep // 2
    right = keep - left
    omitted = max(len(value) - left - right, 0)
    marker = marker_template.format(omitted=omitted)
    tail = value[-right:] if right else ""
    return value[:left] + marker + tail, True, omitted


def request_json(url, *, headers=None, timeout=300, retries=5):
    for attempt in range(retries):
        request = urllib.request.Request(url, headers=dict(headers or {}))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            response_headers = dict(exc.headers or {})
            if (
                retryable_http_error(exc.code, response_headers, body)
                and attempt + 1 < retries
            ):
                wait = sleep_for_rate_limit(response_headers, attempt)
                print(f"  HTTP {exc.code}; sleeping {wait:.0f}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise HttpStatusError(exc.code, url, body, response_headers) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt + 1 >= retries:
                raise ExtractionError(f"request failed for {url}: {exc}") from exc
            time.sleep(min(2 ** attempt, 30))
    raise ExtractionError(f"request retries exhausted for {url}")


def github_json(path, params=None):
    url = GITHUB_API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return request_json(url, headers=github_headers("researchtrails/extraction"))


def github_paginate(path, params=None):
    values = []
    page = 1
    while True:
        query = dict(params or {})
        query.update({"per_page": 100, "page": page})
        chunk = github_json(path, query)
        if not isinstance(chunk, list):
            raise ExtractionError(f"expected a list from GitHub endpoint {path}")
        values.extend(chunk)
        if len(chunk) < 100:
            return values
        page += 1


_last_arxiv_request_at = 0.0


def wait_for_arxiv_slot():
    global _last_arxiv_request_at
    interval = max(float(os.environ.get("ARXIV_MIN_INTERVAL", "3")), 0)
    elapsed = time.monotonic() - _last_arxiv_request_at
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_arxiv_request_at = time.monotonic()


def fetch_arxiv_record(arxiv_id, source="api"):
    """Title, abstract and upload date of version 1, never a later revision."""
    normalized = normalize_arxiv_id(arxiv_id)
    query = urllib.parse.urlencode({"id_list": f"{normalized}v1", "max_results": 1})
    url = f"{arxiv_abs_url(normalized)}v1" if source == "abs" else f"{ARXIV_API}?{query}"
    retries = max(int(os.environ.get("ARXIV_RETRIES", "6")), 1)
    retry_base = max(float(os.environ.get("ARXIV_RETRY_BASE", "5")), 1)
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": os.environ.get(
                    "ARXIV_USER_AGENT", "researchtrails/1.0"
                )
            },
        )
        try:
            wait_for_arxiv_slot()
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            response_headers = dict(exc.headers or {})
            if exc.code in {429, 500, 502, 503, 504} and attempt + 1 < retries:
                wait = max(
                    sleep_for_rate_limit(response_headers, attempt),
                    min(retry_base * (2 ** attempt), 120),
                )
                print(
                    f"  arXiv HTTP {exc.code}; sleeping {wait:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
                continue
            raise ExtractionError(
                f"could not fetch arXiv metadata for {normalized}: "
                f"HTTP {exc.code}: {body[:300]}"
                + (". Retry with --arxiv-metadata-source abs --retry-errors."
                   if exc.code == 406 and source == "api" else "")
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt + 1 < retries:
                wait = min(retry_base * (2 ** attempt), 120)
                print(
                    f"  arXiv request failed; sleeping {wait:.0f}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
                continue
            raise ExtractionError(
                f"could not fetch arXiv metadata for {normalized}: {exc}"
            ) from exc
    try:
        if source == "abs":
            page = data.decode("utf-8")
            block = re.search(
                r'<blockquote\b[^>]*class=["\'][^"\']*\babstract\b[^"\']*["\'][^>]*>'
                r'(.*?)</blockquote>', page, flags=re.S | re.I,
            )
            abstract = html.unescape(re.sub(r"<[^>]*>", "", block.group(1))) if block else ""
            abstract = re.sub(r"^\s*Abstract:\s*", "", abstract)
            title_tag = re.search(
                r'<meta\s+name=["\']citation_title["\']\s+content=["\']([^"\']*)["\']', page, flags=re.I,
            )
            title = html.unescape(title_tag.group(1)) if title_tag else ""
            published = arxiv_original_date(page)
        else:
            root = ET.fromstring(data)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            entry = root.find("atom:entry", namespace)
            if entry is None:
                raise ExtractionError(f"arXiv returned no record for {normalized}")
            title = entry.findtext("atom:title", default="", namespaces=namespace)
            abstract = entry.findtext("atom:summary", default="", namespaces=namespace)
            published = entry.findtext("atom:published", default="", namespaces=namespace)
    except (ValueError, ET.ParseError) as exc:
        raise ExtractionError(
            f"could not parse arXiv metadata for {normalized}: {exc}"
        ) from exc
    title = re.sub(r"\s+", " ", title).strip()
    abstract = re.sub(r"\s+", " ", abstract).strip()
    if not title or not abstract or not published:
        raise ExtractionError(f"arXiv record {normalized}v1 lacks a title, abstract or upload date")
    return {
        "arxiv_id": normalized,
        "title": title,
        "abstract": abstract,
        "published": published[:10],
    }


def run_command(command, *, timeout=600, check=True, env=None, input=None):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=env,
            input=input,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExtractionError(
            f"command failed: {' '.join(map(str, command))}: {exc}"
        ) from exc
    if check and result.returncode:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        raise ExtractionError(
            f"command exited {result.returncode}: {' '.join(map(str, command))}: {stderr}"
        )
    return result


def git(repo_dir, *args, timeout=600, check=True, text=True):
    result = run_command(
        ["git", f"--git-dir={repo_dir}", *map(str, args)],
        timeout=timeout,
        check=check,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    return result.stdout.decode("utf-8", "replace") if text else result.stdout


def object_exists(repo_dir, sha):
    result = run_command(
        ["git", f"--git-dir={repo_dir}", "cat-file", "-e", f"{sha}^{{commit}}"],
        check=False,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    return result.returncode == 0


def prepare_repo_cache(repo_url, owner, name, cache_root, blob_limit_bytes, refresh=True):
    cache_root.mkdir(parents=True, exist_ok=True)
    desired_filter = f"blob:limit={blob_limit_bytes}"
    repo_dir = cache_root / (
        f"{path_slug(owner)}--{path_slug(name)}--blob-limit-{blob_limit_bytes}.git"
    )
    if not repo_dir.exists():
        print(
            f"  cloning history for {owner}/{name} "
            f"(blobs smaller than {blob_limit_bytes} bytes)",
            flush=True,
        )
        run_command(
            [
                "git",
                "clone",
                "--bare",
                f"--filter={desired_filter}",
                repo_url,
                str(repo_dir),
            ],
            timeout=3600,
        )
    elif refresh:
        print(f"  refreshing {owner}/{name}", flush=True)
        run_command(
            [
                "git",
                f"--git-dir={repo_dir}",
                "fetch",
                "--prune",
                "origin",
                "+refs/heads/*:refs/heads/*",
            ],
            timeout=3600,
        )
    return repo_dir


def repo_default_branch(repo_dir):
    branch = git(repo_dir, "symbolic-ref", "--short", "HEAD", check=False).strip()
    branch = branch.removeprefix("refs/heads/")
    if not branch or not object_exists(repo_dir, f"refs/heads/{branch}"):
        raise ExtractionError("repository does not identify a default branch")
    return branch


def default_branch_commits(repo_dir, branch, cutoff):
    output = git(
        repo_dir,
        "rev-list",
        "--reverse",
        # Commit date order, not graph order: a merged side branch must not place
        # months-old commits after the mainline commits that came before it.
        "--date-order",
        f"--until={cutoff}",
        f"refs/heads/{branch}",
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def trim_text(value, max_chars):
    value = value or ""
    if len(value) <= max_chars:
        return value, False
    marker = f"\n[truncated; original length={len(value)} characters]\n"
    keep = max(max_chars - len(marker), 0)
    left = keep // 2
    right = keep - left
    tail = value[-right:] if right else ""
    return value[:left] + marker + tail, True


def discover_pull_requests(owner, name, default_shas, cutoff, enabled=True):
    if not enabled:
        return {}, [], ["Pull-request enrichment disabled by --skip-prs."]
    by_number = {}
    commit_to_prs = {}
    limitations = []
    total = len(default_shas)
    for index, sha in enumerate(default_shas, 1):
        try:
            pulls = github_json(f"/repos/{owner}/{name}/commits/{sha}/pulls")
        except ExtractionError as exc:
            limitations.append(f"Could not inspect associated PRs for {sha}: {exc}")
            continue
        accepted = []
        for pull in pulls if isinstance(pulls, list) else []:
            merged_at = pull.get("merged_at")
            if not merged_at or merged_at > cutoff:
                continue
            number = int(pull["number"])
            by_number[number] = pull
            accepted.append(number)
        if accepted:
            commit_to_prs[sha] = sorted(set(accepted))
        if total >= 25 and (index % 25 == 0 or index == total):
            print(f"  PR lookup {index}/{total} ({len(by_number)} merged PRs)", flush=True)

    pr_records = []
    for number, pull in sorted(by_number.items()):
        try:
            commits = github_paginate(f"/repos/{owner}/{name}/pulls/{number}/commits")
        except ExtractionError as exc:
            limitations.append(f"Could not list commits for PR #{number}: {exc}")
            commits = []
        body, body_truncated = trim_text(pull.get("body") or "", 8000)
        record = {
            "number": number,
            "title": pull.get("title") or "",
            "body": body,
            "body_truncated": body_truncated,
            "html_url": pull.get("html_url") or "",
            "created_at": pull.get("created_at") or "",
            "merged_at": pull.get("merged_at") or "",
            "merge_commit_sha": pull.get("merge_commit_sha") or "",
            "base_branch": (pull.get("base") or {}).get("ref") or "",
            "head_branch": (pull.get("head") or {}).get("ref") or "",
            "commit_shas": [item.get("sha", "") for item in commits if item.get("sha")],
        }
        pr_records.append(record)
        for sha in record["commit_shas"]:
            commit_to_prs.setdefault(sha, []).append(number)
    return commit_to_prs, pr_records, limitations


def fetch_pull_refs(repo_dir, pr_records):
    limitations = []
    for pull in pr_records:
        number = pull["number"]
        missing = [sha for sha in pull["commit_shas"] if not object_exists(repo_dir, sha)]
        if not missing:
            continue
        result = run_command(
            [
                "git",
                f"--git-dir={repo_dir}",
                "fetch",
                "origin",
                f"+refs/pull/{number}/head:refs/pull/{number}/head",
            ],
            timeout=1800,
            check=False,
        )
        if result.returncode:
            limitations.append(
                f"PR #{number} branch commits were reported by GitHub but its retained "
                "ref could not be fetched."
            )
    return limitations


def chronological_commits(default_shas, pr_records, repo_dir, cutoff):
    default_set = set(default_shas)
    extras_by_anchor = {}
    limitations = []
    cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    for pull in pr_records:
        merge_sha = pull.get("merge_commit_sha")
        anchor = merge_sha if merge_sha in default_set else None
        if anchor is None:
            candidates = [sha for sha in pull["commit_shas"] if sha in default_set]
            anchor = candidates[-1] if candidates else None
        if anchor is None:
            limitations.append(
                f"PR #{pull['number']} has no recoverable default-branch anchor commit."
            )
            continue
        for sha in pull["commit_shas"]:
            if sha in default_set or not object_exists(repo_dir, sha):
                continue
            date = commit_metadata(repo_dir, sha)["date"]
            try:
                parsed = datetime.fromisoformat(date.replace("Z", "+00:00"))
            except ValueError:
                limitations.append(f"PR commit {sha} has an unparsable date {date!r}.")
                continue
            if parsed <= cutoff_dt:
                extras_by_anchor.setdefault(anchor, []).append(sha)
    sequence = []
    seen = set()
    for sha in default_shas:
        for extra in extras_by_anchor.get(sha, []):
            if extra not in seen:
                sequence.append(extra)
                seen.add(extra)
        if sha not in seen:
            sequence.append(sha)
            seen.add(sha)
    return sequence, limitations


def commit_metadata(repo_dir, sha):
    raw = git(
        repo_dir,
        "log",
        "-1",
        "--no-patch",
        "--format=%H%x00%cI%x00%an%x00%B",
        sha,
    )
    parts = raw.split("\x00", 3)
    if len(parts) != 4:
        raise ExtractionError(f"could not parse metadata for commit {sha}")
    parent_line = git(repo_dir, "rev-list", "--parents", "-n", "1", sha).strip().split()
    return {
        "sha": parts[0].strip(),
        "date": parts[1].strip(),
        "author": parts[2].strip(),
        "message": parts[3].strip(),
        "parents": parent_line[1:],
    }


def parse_name_status(raw):
    parts = raw.split(b"\x00")
    if parts and not parts[-1]:
        parts.pop()
    files = []
    index = 0
    while index < len(parts):
        status = parts[index].decode("utf-8", "replace")
        index += 1
        if index >= len(parts):
            break
        old_path = ""
        path = parts[index].decode("utf-8", "replace")
        index += 1
        if status.startswith(("R", "C")) and index < len(parts):
            old_path = path
            path = parts[index].decode("utf-8", "replace")
            index += 1
        files.append({"status": status, "path": path, "old_path": old_path})
    return files


def parse_numstat(raw):
    stats = {}
    for item in raw.split(b"\x00"):
        if not item:
            continue
        fields = item.split(b"\t", 2)
        if len(fields) != 3:
            continue
        added_raw, deleted_raw, path_raw = fields
        path = path_raw.decode("utf-8", "replace")
        binary = added_raw == b"-" or deleted_raw == b"-"
        stats[path] = {
            "additions": None if binary else int(added_raw),
            "deletions": None if binary else int(deleted_raw),
            "binary": binary,
        }
    return stats


def changed_files(repo_dir, metadata):
    sha = metadata["sha"]
    if metadata["parents"]:
        base = [metadata["parents"][0], sha]
        raw = git(
            repo_dir,
            "diff",
            "--name-status",
            "-z",
            "--find-renames=100%",
            *base,
            text=False,
        )
        numstat_raw = git(
            repo_dir,
            "diff",
            "--numstat",
            "-z",
            "--find-renames=100%",
            *base,
            text=False,
            check=False,
        )
    else:
        raw = git(
            repo_dir,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            "--find-renames=100%",
            sha,
            text=False,
        )
        numstat_raw = git(
            repo_dir,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--numstat",
            "-r",
            "-z",
            "--find-renames=100%",
            sha,
            text=False,
            check=False,
        )
    stats = parse_numstat(numstat_raw)
    files = parse_name_status(raw)
    for changed in files:
        changed.update(
            stats.get(
                changed["path"],
                {"additions": None, "deletions": None, "binary": False},
            )
        )
    return files


def missing_blob_oids(repo_dir):
    output = git(repo_dir, "rev-list", "--objects", "--missing=print", "--all")
    return {
        line[1:].split(None, 1)[0]
        for line in output.splitlines()
        if line.startswith("?")
    }


def file_metadata_for_commit(repo_dir, metadata, files, missing_oids):
    """Read each side's tree once and query changed blobs' sizes in one batch."""
    parent = metadata["parents"][0] if metadata["parents"] else ""
    sides = []
    for label, revision, paths in (
        ("parent", parent, {item.get("old_path") or item["path"] for item in files}),
        ("current", metadata["sha"], {item["path"] for item in files}),
    ):
        entries = {}
        if revision and paths:
            if len(files) >= 1000:
                print(f"    metadata: reading {label} tree", flush=True)
            # Include directory entries for file/submodule-to-directory changes.
            raw = git(repo_dir, "ls-tree", "-r", "-t", "-z", revision, text=False)
            for record in raw.split(b"\x00"):
                if not record:
                    continue
                header, path_raw = record.split(b"\t", 1)
                path = path_raw.decode("utf-8", "replace")
                if path in paths:
                    _mode, kind, oid = header.decode("ascii").split()
                    entries[path] = {"type": kind, "oid": oid}
        sides.append(entries)

    oids = sorted({
        entry["oid"]
        for entries in sides
        for entry in entries.values()
        if entry["type"] == "blob" and entry["oid"] not in missing_oids
    })
    sizes = {}
    if oids:
        if len(files) >= 1000:
            print(f"    metadata: checking {len(oids)} blob sizes in one batch", flush=True)
        result = run_command(
            [
                "git", f"--git-dir={repo_dir}", "cat-file",
                "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            ],
            input=("\n".join(oids) + "\n").encode("ascii"),
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        lines = result.stdout.decode("ascii").splitlines()
        if len(lines) != len(oids):
            raise ExtractionError("git cat-file returned incomplete batch metadata")
        for oid, line in zip(oids, lines):
            fields = line.split()
            if fields == [oid, "missing"]:
                sizes[oid] = None
            elif (
                len(fields) == 3
                and fields[:2] == [oid, "blob"]
                and fields[2].isdigit()
            ):
                sizes[oid] = int(fields[2])
            else:
                raise ExtractionError(f"invalid git blob metadata: {line!r}")
    for entries in sides:
        for entry in entries.values():
            if entry["type"] == "blob":
                entry["size"] = (
                    None if entry["oid"] in missing_oids else sizes[entry["oid"]]
                )
    return sides


def file_side_info(repo_dir, entry, missing_oids, *, inspect_binary=False):
    def side(note, *, size=None, binary=False):
        return {"size": size, "note": note, "binary": binary}

    absent = "file does not exist at this side of the change"
    if not entry:
        return side(absent)
    if entry["type"] != "blob":
        return side(f"historical tree entry has type {entry['type']}")
    if entry["oid"] in missing_oids:
        return side("historical blob is unavailable under the configured clone limit")
    size = entry["size"]
    if size is None:
        return side("historical blob is unavailable in the local cache")
    binary = False
    if inspect_binary:
        raw = git(repo_dir, "cat-file", "blob", entry["oid"], text=False)
        binary = b"\x00" in raw[:8192]
    return side("", size=size, binary=binary)


def raw_file_content(repo_dir, entry, max_bytes, missing_oids):
    info = file_side_info(repo_dir, entry, missing_oids)
    if info["size"] is None:
        return None, info["note"]
    if info["size"] > max_bytes:
        return None, f"historical file content omitted because it is {info['size']} bytes"
    return git(repo_dir, "cat-file", "blob", entry["oid"], text=False), ""


def decode_text_blob(raw):
    if raw is None:
        return None
    if b"\x00" in raw[:8192]:
        raise ExtractionError("file classified as text contains NUL bytes")
    return raw.decode("utf-8", "replace")


def file_patch(repo_dir, metadata, changed, before_entry, after_entry, max_blob_bytes):
    sha = metadata["sha"]
    paths = [path for path in [changed.get("old_path"), changed["path"]] if path]
    missing_paths = []
    oversized_paths = {}
    for path, entry in (
        (changed.get("old_path") or changed["path"], before_entry),
        (changed["path"], after_entry),
    ):
        if not entry or entry["type"] != "blob":
            continue
        size = entry["size"]
        if size is None:
            missing_paths.append(path)
        elif size > max_blob_bytes:
            oversized_paths[path] = max(oversized_paths.get(path, 0), size)
    if missing_paths:
        return "", (
            "patch omitted because an involved blob is unavailable under the configured "
            "clone limit: "
            + ", ".join(sorted(set(missing_paths)))
        )
    if oversized_paths:
        details = ", ".join(
            f"{path} ({size} bytes)" for path, size in sorted(oversized_paths.items())
        )
        return "", (
            f"patch omitted because an involved blob exceeds the {max_blob_bytes}-byte "
            f"patch limit: {details}"
        )
    # A submodule/file replaced by a directory shares its path with added children.
    # Keep the recorded change type so those children get only their own patches.
    diff_filter = f"--diff-filter={changed['status'][0]}"
    if metadata["parents"]:
        patch = git(
            repo_dir,
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--find-renames=100%",
            "--unified=3",
            diff_filter,
            metadata["parents"][0],
            sha,
            "--",
            *paths,
        )
    else:
        patch = git(
            repo_dir,
            "show",
            "--format=",
            "--no-color",
            "--no-ext-diff",
            "--find-renames=100%",
            "--unified=3",
            diff_filter,
            sha,
            "--",
            *paths,
        )
    return patch, ""


def split_exact_text(value, max_chars):
    if not value:
        return [""]
    chunks = []
    start = 0
    while start < len(value):
        end = min(start + max_chars, len(value))
        if end < len(value):
            newline = value.rfind("\n", start, end)
            if newline > start:
                end = newline + 1
        chunks.append(value[start:end])
        start = end
    return chunks


def extract_evidence_for_commit(
    repo_dir,
    metadata,
    sequence_index,
    pr_numbers,
    evidence_start,
    patch_chunk_chars,
    missing_oids,
    seen_paths,
    creation_limits,
    patch_blob_limit=DEFAULT_GIT_BLOB_LIMIT,
    max_file_patch_chars=32000,
    text_preview_chars=6000,
):
    records = []
    evidence_number = evidence_start
    label = f"commit {sequence_index} {metadata['sha'][:12]}"
    started = time.monotonic()
    print(f"  {label}: listing changed files and counting lines", flush=True)
    files = changed_files(repo_dir, metadata)
    total_files = len(files)
    if total_files >= 1000:
        print(f"  {label}: loading metadata for {total_files} changed files", flush=True)
    before_entries, after_entries = file_metadata_for_commit(
        repo_dir, metadata, files, missing_oids
    )
    if total_files >= 1000:
        print(f"  {label}: files 0/{total_files}", flush=True)
    last_progress = time.monotonic()
    parent = metadata["parents"][0] if metadata["parents"] else ""
    for file_index, changed in enumerate(files, 1):
        path = changed["path"]
        old_path = changed.get("old_path") or path
        before_entry = before_entries.get(old_path)
        after_entry = after_entries.get(path)
        initial_policy = classify_file(path, binary=bool(changed.get("binary")))
        inspect_binary = (
            initial_policy.mode != "manifest_only"
            and changed.get("additions") is None
            and changed.get("deletions") is None
        )
        before_info = file_side_info(
            repo_dir, before_entry, missing_oids, inspect_binary=inspect_binary
        )
        after_info = file_side_info(
            repo_dir,
            after_entry,
            missing_oids,
            inspect_binary=inspect_binary,
        )
        binary = bool(
            changed.get("binary")
            or before_info.get("binary")
            or after_info.get("binary")
        )
        policy = classify_file(path, binary=binary)
        status = changed["status"][:1]
        first_seen = path not in seen_paths
        creation_snapshot = bool(
            policy.read_at_creation and first_seen and status == "A"
        )

        full_text = ""
        omitted_reason = ""
        evidence_kind = "manifest"
        applied_creation_limit = 0

        if policy.mode == "manifest_only":
            unavailable = [
                info["note"]
                for info in (before_info, after_info)
                if "unavailable" in info.get("note", "")
            ]
            omitted_reason = policy.reason
            if unavailable:
                omitted_reason += "; " + "; ".join(sorted(set(unavailable)))
        elif policy.mode == "notebook_source":
            if creation_snapshot:
                after_raw, note = raw_file_content(
                    repo_dir, after_entry, patch_blob_limit, missing_oids
                )
                if after_raw is None:
                    omitted_reason = f"notebook creation snapshot unavailable: {note}"
                else:
                    try:
                        full_text = normalize_notebook(after_raw)
                        evidence_kind = "creation_snapshot"
                    except NotebookFormatError as exc:
                        policy = FilePolicy(
                            "unparsed_notebook",
                            "text_preview",
                            False,
                            "",
                            f"notebook normalization failed ({exc}); bounded raw diff preview retained",
                        )
                        full_text, omitted_reason = file_patch(
                            repo_dir, metadata, changed, before_entry, after_entry,
                            patch_blob_limit,
                        )
                        evidence_kind = "text_preview"
            else:
                before_raw, before_note = raw_file_content(
                    repo_dir, before_entry, patch_blob_limit, missing_oids
                )
                after_raw, after_note = raw_file_content(
                    repo_dir, after_entry, patch_blob_limit, missing_oids
                )
                unavailable = []
                if parent and status != "A" and before_raw is None:
                    unavailable.append(before_note)
                if status != "D" and after_raw is None:
                    unavailable.append(after_note)
                if unavailable:
                    omitted_reason = (
                        "normalized notebook diff unavailable: "
                        + "; ".join(sorted(set(unavailable)))
                    )
                else:
                    try:
                        full_text = notebook_source_diff(
                            before_raw, after_raw, old_path, path
                        )
                        evidence_kind = "diff"
                        if not full_text:
                            omitted_reason = (
                                "notebook source cells did not change after stripping "
                                "outputs and execution metadata"
                            )
                    except NotebookFormatError as exc:
                        policy = FilePolicy(
                            "unparsed_notebook",
                            "text_preview",
                            False,
                            "",
                            f"notebook normalization failed ({exc}); bounded raw diff preview retained",
                        )
                        full_text, omitted_reason = file_patch(
                            repo_dir, metadata, changed, before_entry, after_entry,
                            patch_blob_limit,
                        )
                        evidence_kind = "text_preview"
        elif policy.mode == "source_patch" and creation_snapshot:
            after_raw, note = raw_file_content(
                repo_dir, after_entry, patch_blob_limit, missing_oids
            )
            if after_raw is None:
                omitted_reason = f"creation snapshot unavailable: {note}"
            else:
                try:
                    full_text = decode_text_blob(after_raw)
                    evidence_kind = "creation_snapshot"
                except ExtractionError as exc:
                    omitted_reason = f"creation snapshot unavailable: {exc}"
        else:
            full_text, omitted_reason = file_patch(
                repo_dir, metadata, changed, before_entry, after_entry, patch_blob_limit
            )
            evidence_kind = (
                "diff" if policy.mode == "source_patch" else "text_preview"
            )

        if evidence_kind == "creation_snapshot":
            applied_creation_limit = creation_limits[policy.creation_limit_kind]
            text_limit = applied_creation_limit
        elif policy.mode in {"source_patch", "notebook_source"}:
            text_limit = max_file_patch_chars
        else:
            text_limit = text_preview_chars
        retained_text, truncated, removed_chars = limit_text(full_text, text_limit)
        note = ""
        if truncated:
            note = (
                f"{evidence_kind} reduced from {len(full_text)} to "
                f"{len(retained_text)} characters; {removed_chars} characters omitted"
            )
        chunks = split_exact_text(retained_text, patch_chunk_chars)
        for part, chunk in enumerate(chunks, 1):
            record = {
                "evidence_id": f"E{evidence_number:07d}",
                "commit_sha": metadata["sha"],
                "date": metadata["date"],
                "path": path,
                "old_path": changed.get("old_path") or "",
                "change_status": changed["status"],
                "additions": changed.get("additions"),
                "deletions": changed.get("deletions"),
                "binary_detected": binary,
                "before_bytes": before_info.get("size"),
                "after_bytes": after_info.get("size"),
                "policy_category": policy.category,
                "evidence_mode": policy.mode,
                "evidence_kind": evidence_kind,
                "read_at_creation": policy.read_at_creation,
                "read_at_creation_applied": creation_snapshot,
                "creation_limit_kind": policy.creation_limit_kind,
                "creation_limit_chars": applied_creation_limit,
                "selection_reason": policy.reason,
                "patch_part": part,
                "patch_parts": len(chunks),
                "patch": chunk,
                "patch_omitted_reason": omitted_reason,
                "patch_note": note,
                "patch_original_chars": len(full_text),
                "patch_truncated": truncated,
            }
            records.append(record)
            evidence_number += 1
        seen_paths.add(path)
        now = time.monotonic()
        if now - last_progress >= 10 or (
            total_files >= 1000 and file_index == total_files
        ):
            print(
                f"  {label}: files {file_index}/{total_files} "
                f"({len(records)} records, {now - started:.1f}s elapsed)",
                flush=True,
            )
            last_progress = now
    commit = {
        **metadata,
        "sequence_index": sequence_index,
        "pull_requests": sorted(set(pr_numbers)),
        "evidence_ids": [record["evidence_id"] for record in records],
    }
    return commit, records, evidence_number


def build_evidence(
    row,
    repo_dir,
    owner,
    name,
    default_branch,
    cutoff,
    include_prs,
    patch_chunk_chars,
    git_blob_limit,
    max_file_patch_chars,
    text_preview_chars,
    creation_limits,
):
    default_shas = default_branch_commits(repo_dir, default_branch, cutoff)
    if not default_shas:
        raise ExtractionError("default branch has no commits at or before the paper cutoff")
    commit_to_prs, pr_records, limitations = discover_pull_requests(
        owner, name, default_shas, cutoff, enabled=include_prs
    )
    limitations.extend(fetch_pull_refs(repo_dir, pr_records))
    sequence, sequence_limitations = chronological_commits(
        default_shas, pr_records, repo_dir, cutoff
    )
    limitations.extend(sequence_limitations)
    missing_oids = missing_blob_oids(repo_dir)
    if missing_oids:
        limitations.append(
            f"{len(missing_oids)} oversized historical blobs were intentionally not "
            f"downloaded (clone limit {git_blob_limit} bytes)."
        )
    commits = []
    evidence_records = []
    seen_paths = set()
    next_evidence = 1
    total = len(sequence)
    print(f"  evidence extraction 0/{total} commits", flush=True)
    for index, sha in enumerate(sequence, 1):
        metadata = commit_metadata(repo_dir, sha)
        commit, records, next_evidence = extract_evidence_for_commit(
            repo_dir,
            metadata,
            index,
            commit_to_prs.get(sha, []),
            next_evidence,
            patch_chunk_chars,
            missing_oids,
            seen_paths,
            creation_limits,
            git_blob_limit,
            max_file_patch_chars,
            text_preview_chars,
        )
        commits.append(commit)
        evidence_records.extend(records)
        if total >= 10 and (index % 10 == 0 or index == total):
            print(
                f"  evidence {index}/{total} commits ({len(evidence_records)} records)",
                flush=True,
            )
    selection_summary = summarize_selection(evidence_records)
    print(
        "  evidence policy: "
        f"{selection_summary['changed_files']} changed files, "
        f"{selection_summary['creation_snapshots']} creation snapshots, "
        f"{selection_summary['manifest_only_files']} manifest-only, "
        f"{selection_summary['patch_retained_chars']} retained characters",
        flush=True,
    )
    return {
        "created_at": utc_now(),
        "source": {
            "arxiv": f"https://arxiv.org/abs/{normalize_arxiv_id(row['arxiv_id'])}",
            "github": row["repo_url"],
        },
        "paper": {
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "paper_date": row.get("paper_date", ""),
            "cutoff": cutoff,
        },
        "repository": {
            "owner": owner,
            "name": name,
            "default_branch": default_branch,
            "default_branch_tip_at_scan": git(
                repo_dir, "rev-parse", f"refs/heads/{default_branch}"
            ).strip(),
            "default_branch_commits": len(default_shas),
            "commits_scanned": len(commits),
        },
        "extraction": {
            "include_pull_requests": include_prs,
            "patch_chunk_chars": patch_chunk_chars,
            "git_blob_limit": git_blob_limit,
            "max_file_patch_chars": max_file_patch_chars,
            "text_preview_chars": text_preview_chars,
            "creation_limits": creation_limits,
            "oversized_blobs_omitted": len(missing_oids),
            "patches_omitted": sum(
                record.get("patch_part") == 1
                and bool(record.get("patch_omitted_reason"))
                for record in evidence_records
            ),
        },
        "selection_summary": selection_summary,
        "pull_requests": pr_records,
        "commits": commits,
        "evidence_records": evidence_records,
        "limitations": limitations,
    }


def apply_arxiv_v1(row, source="api"):
    """Take the paper's title, abstract and upload date from arXiv version 1."""
    record = fetch_arxiv_record(row.get("arxiv_id"), source=source)
    row["title"] = record["title"]
    row["abstract"] = record["abstract"]
    row["paper_date"] = record["published"]


def summarize_selection(records):
    files = [record for record in records if record.get("patch_part") == 1]
    categories = Counter(record.get("policy_category", "unknown") for record in files)
    modes = Counter(record.get("evidence_mode", "unknown") for record in files)
    kinds = Counter(record.get("evidence_kind", "unknown") for record in files)
    original_patch_chars = sum(
        int(record.get("patch_original_chars") or 0) for record in files
    )
    retained_patch_chars = sum(len(record.get("patch") or "") for record in records)
    return {
        "changed_files": len(files),
        "content_included_files": sum(
            bool(record.get("patch"))
            for record in files
        ),
        "read_at_creation_files": sum(
            bool(record.get("read_at_creation")) for record in files
        ),
        "creation_snapshots": kinds.get("creation_snapshot", 0),
        "manifest_only_files": modes.get("manifest_only", 0),
        "truncated_patch_files": sum(bool(record.get("patch_truncated")) for record in files),
        "patch_original_chars": original_patch_chars,
        "patch_retained_chars": retained_patch_chars,
        "patch_removed_chars": max(original_patch_chars - retained_patch_chars, 0),
        "files_by_category": dict(sorted(categories.items())),
        "files_by_mode": dict(sorted(modes.items())),
        "files_by_kind": dict(sorted(kinds.items())),
    }


def audit_evidence_files(evidence_paths):
    groups = {}
    projects = set()
    evidence_file_count = 0
    record_count = 0
    for evidence_path in evidence_paths:
        evidence_path = Path(evidence_path)
        with evidence_path.open() as handle:
            evidence = json.load(handle)
        evidence_file_count += 1
        repo = evidence.get("source", {}).get("github") or evidence_path.parent.name
        projects.add(repo)
        records = evidence.get("evidence_records", [])
        record_count += len(records)
        for record in records:
            suffix = extension(record.get("path")) or "<none>"
            category = record.get("policy_category", "unknown")
            mode = record.get("evidence_mode", "unknown")
            kind = record.get("evidence_kind", "unknown")
            key = (suffix, category, mode, kind)
            group = groups.setdefault(
                key,
                {
                    "extension": suffix,
                    "category": category,
                    "mode": mode,
                    "kind": kind,
                    "repositories": set(),
                    "commits": set(),
                    "changed_files": 0,
                    "binary_files": 0,
                    "file_bytes": 0,
                    "patch_original_chars": 0,
                    "patch_retained_chars": 0,
                    "sample_paths": [],
                },
            )
            group["patch_retained_chars"] += len(record.get("patch") or "")
            if record.get("patch_part") != 1:
                continue
            group["repositories"].add(repo)
            group["commits"].add(f"{repo}\x00{record.get('commit_sha', '')}")
            group["changed_files"] += 1
            group["binary_files"] += int(bool(record.get("binary_detected")))
            group["file_bytes"] += max(
                int(record.get("before_bytes") or 0),
                int(record.get("after_bytes") or 0),
            )
            group["patch_original_chars"] += int(
                record.get("patch_original_chars") or 0
            )
            path = str(record.get("path") or "")
            if path and path not in group["sample_paths"] and len(group["sample_paths"]) < 8:
                group["sample_paths"].append(path)

    rows = []
    for group in groups.values():
        row = dict(group)
        row["repository_count"] = len(row.pop("repositories"))
        row["commit_count"] = len(row.pop("commits"))
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["file_bytes"],
            -row["patch_original_chars"],
            row["extension"],
            row["category"],
        )
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_files_scanned": evidence_file_count,
        "projects_scanned": len(projects),
        "records_scanned": record_count,
        "changed_files": sum(row["changed_files"] for row in rows),
        "groups": rows,
    }


def extraction_status_counts(rows):
    counts = Counter(row.get("evidence_status") or "pending" for row in rows)
    return (
        f"prepared={counts['prepared']} error={counts['error']} "
        f"pending={counts['pending']}"
    )


def select_progressive_rows(progressive, indices, offset, limit):
    if indices is None:
        end = None if limit is None else offset + limit
        return progressive[offset:end]

    by_index = {}
    for row in progressive:
        try:
            index = int(row["idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExtractionError("progressive row has a missing or invalid idx") from exc
        if index in by_index:
            raise ExtractionError(f"progressive idx {index} occurs more than once")
        by_index[index] = row
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ExtractionError(
            "requested indices are not progressive rows: "
            + ", ".join(str(index) for index in missing)
        )
    return [by_index[index] for index in indices]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue", default=os.environ.get("VENUE", "neurips"))
    parser.add_argument("--year", type=int, default=int(os.environ.get("YEAR", "2025")))
    parser.add_argument(
        "--arxiv-metadata-source", choices=["api", "abs"], default="api",
        help="fetch the version-1 title, abstract and upload date from the Atom API or the v1 abstract page",
    )
    parser.add_argument("--in", dest="input_csv", default=os.environ.get("EXTRACTION_IN"))
    parser.add_argument("--annotations-dir", default=os.environ.get("ANNOTATIONS_DIR"))
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("EXTRACTION_CACHE_DIR"),
    )
    parser.add_argument("--audit-out", default=os.environ.get("EXTRACTION_AUDIT_OUT"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        help="exact progressive-row idx values to process, in the supplied order",
    )
    parser.add_argument(
        "--patch-chunk-chars",
        type=int,
        default=int(os.environ.get("EXTRACTION_PATCH_CHARS", "16000")),
    )
    parser.add_argument(
        "--max-file-patch-chars",
        type=int,
        default=int(os.environ.get("EXTRACTION_MAX_FILE_PATCH_CHARS", "32000")),
    )
    parser.add_argument(
        "--text-preview-chars",
        type=int,
        default=int(os.environ.get("EXTRACTION_TEXT_PREVIEW_CHARS", "6000")),
    )
    for kind, limit in DEFAULT_CREATION_LIMITS.items():
        parser.add_argument(
            f"--{kind}-creation-chars",
            type=int,
            default=int(
                os.environ.get(
                    f"EXTRACTION_{kind.upper()}_CREATION_CHARS", str(limit)
                )
            ),
        )
    parser.add_argument(
        "--git-blob-limit",
        type=int,
        default=DEFAULT_GIT_BLOB_LIMIT,
    )
    parser.add_argument("--sleep", type=float, default=0)
    parser.add_argument("--skip-prs", action="store_true")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args(argv)

    prefix = f"{venue_slug(args.venue)}_{args.year}"
    args.input_csv = resolve_path(args.input_csv, BASE / f"{prefix}_filtered.csv")
    args.annotations_dir = resolve_path(
        args.annotations_dir, REPO_ROOT / "annotations" / prefix
    )
    args.cache_dir = resolve_path(
        args.cache_dir, BASE / ".cache" / "extraction" / "repos"
    )
    args.audit_out = resolve_path(
        args.audit_out, BASE / f"{prefix}_file_type_audit.json"
    )
    positive = [
        "patch_chunk_chars",
        "max_file_patch_chars",
        "text_preview_chars",
        "git_blob_limit",
    ] + [f"{kind}_creation_chars" for kind in DEFAULT_CREATION_LIMITS]
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.indices is not None:
        if args.offset or args.limit is not None:
            parser.error("--indices cannot be combined with --offset or --limit")
        if len(args.indices) != len(set(args.indices)):
            parser.error("--indices contains duplicate values")
    return args


def main():
    args = parse_args()
    if not args.input_csv.exists():
        sys.exit(f"input CSV does not exist: {args.input_csv}")
    if not args.skip_prs and not os.environ.get("GITHUB_TOKEN"):
        print(
            "warning: GITHUB_TOKEN is not set; PR enrichment will quickly exhaust "
            "the unauthenticated GitHub API budget",
            file=sys.stderr,
        )
    with args.input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        sys.exit(f"input CSV has no rows: {args.input_csv}")
    for addition in CSV_ADDITIONS:
        if addition not in fieldnames:
            fieldnames.append(addition)
    for row in rows:
        for addition in CSV_ADDITIONS:
            row.setdefault(addition, "")

    progressive = [row for row in rows if row.get("heuristic_bucket") == "progressive"]
    try:
        selected = select_progressive_rows(
            progressive, args.indices, args.offset, args.limit
        )
    except ExtractionError as exc:
        sys.exit(str(exc))
    if not selected:
        print("No progressive rows selected.")
        return
    creation_limits = {
        kind: getattr(args, f"{kind}_creation_chars")
        for kind in DEFAULT_CREATION_LIMITS
    }
    args.annotations_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"selected {len(selected)}/{len(progressive)} progressive rows for evidence extraction",
        flush=True,
    )
    for position, row in enumerate(selected, 1):
        if (
            row.get("evidence_status") == "error"
            and not args.retry_errors
            and not args.force
        ):
            print(
                f"[{position}/{len(selected)}] skip {row.get('repo_url')}: evidence error "
                "(use --retry-errors)",
                flush=True,
            )
            continue
        print(
            f"[{position}/{len(selected)}] idx={row.get('idx')} {row.get('repo_url')}",
            flush=True,
        )
        try:
            parsed = parse_repo(row.get("repo_url"))
            if not parsed:
                raise ExtractionError("repo_url is not a GitHub repository URL")
            owner, name = parsed
            project_dir = args.annotations_dir / (
                f"{path_slug(row.get('idx'))}-{path_slug(owner)}-{path_slug(name)}"
            )
            evidence_path = project_dir / "evidence.json"
            project_dir.mkdir(parents=True, exist_ok=True)

            if evidence_path.exists() and not args.force:
                with evidence_path.open() as handle:
                    evidence = json.load(handle)
                print(
                    f"  using existing evidence ({len(evidence['commits'])} commits); "
                    "use --force to rebuild",
                    flush=True,
                )
            else:
                apply_arxiv_v1(row, source=args.arxiv_metadata_source)
                cutoff = paper_month_cutoff(row["paper_date"])
                repo_dir = prepare_repo_cache(
                    row["repo_url"],
                    owner,
                    name,
                    args.cache_dir,
                    args.git_blob_limit,
                    refresh=not args.no_refresh,
                )
                default_branch = repo_default_branch(repo_dir)
                evidence = build_evidence(
                    row,
                    repo_dir,
                    owner,
                    name,
                    default_branch,
                    cutoff,
                    not args.skip_prs,
                    args.patch_chunk_chars,
                    args.git_blob_limit,
                    args.max_file_patch_chars,
                    args.text_preview_chars,
                    creation_limits,
                )
                atomic_write_json(evidence_path, evidence)
            row["evidence_status"] = "prepared"
            row["evidence_error"] = ""
            print(
                f"  result: prepared commits={len(evidence.get('commits', []))} "
                f"records={len(evidence.get('evidence_records', []))}",
                flush=True,
            )
        except KeyboardInterrupt:
            atomic_write_csv(args.input_csv, rows, fieldnames)
            print(
                "\nInterrupted; completed repositories are saved. Rerunning reuses "
                "Git caches but restarts the unfinished repository from its first commit.",
                file=sys.stderr,
            )
            raise
        except Exception as exc:
            row["evidence_status"] = "error"
            row["evidence_error"] = str(exc)
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
        atomic_write_csv(args.input_csv, rows, fieldnames)
        print(
            f"  progress: {position}/{len(selected)} {extraction_status_counts(selected)}",
            flush=True,
        )
        if args.sleep and position < len(selected):
            time.sleep(args.sleep)

    if not args.no_audit:
        evidence_paths = sorted(args.annotations_dir.glob("*/evidence.json"))
        report = audit_evidence_files(evidence_paths)
        atomic_write_json(args.audit_out, report)
        print(
            f"AUDIT -> {args.audit_out} "
            f"({report['projects_scanned']} projects)",
            flush=True,
        )
    print("EVIDENCE ->", args.annotations_dir)


if __name__ == "__main__":
    main()
