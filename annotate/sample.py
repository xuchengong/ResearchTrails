#!/usr/bin/env python3
"""Build a paper->repo sample for one top-conference venue/year.

Workflow:
1. For the given (venue, year), pull the full paper list from the Semantic
   Scholar bulk-search API and order it by publication date.
2. For each paper, get its arXiv id from Semantic Scholar metadata. Collect every
   direct GitHub repository link from the arXiv abstract and from the latest PDF
   before its References/Appendix. Retain a repository only when its README
   contains the paper's full title, case-insensitively. Emit one row per retained
   paper-repository pair.
3. Store selected pairs in [venue]_[year]_buffer.json. A sidecar
   [venue]_[year]_buffer.state.json makes the job resumable.

The output JSON remains a list of selected rows so measure.py can consume it via
BUFFER=<file>.json without special handling.
"""
import argparse
from collections import Counter
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    GITHUB_API,
    arxiv_abs_url,
    atomic_write_json,
    decode_readme_response,
    extract_arxiv_id,
    github_headers,
    rate_limit_reset_wait,
    readme_names_paper,
    venue_slug,
)
from envload import load_env  # noqa: E402

load_env()

BASE = Path(__file__).resolve().parent
S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "paperId,title,externalIds,venue,publicationDate,year,url,openAccessPdf"
UA = {"User-Agent": "sampling-study"}
PDF_MAX_BYTES = int(os.environ.get("ARXIV_PDF_MAX_BYTES", str(30 * 1024 * 1024)))
README_MAX_BYTES = int(os.environ.get("GITHUB_README_MAX_BYTES", str(2 * 1024 * 1024)))
SAMPLE_VERSION = "all-main-text-links-readme-full-title-v2"

SKIP_OWNERS = {
    "about",
    "collections",
    "explore",
    "features",
    "join",
    "login",
    "marketplace",
    "orgs",
    "pricing",
    "settings",
    "sponsors",
    "topics",
}

TRAILING_URL_JUNK = ".,;:!?)>]}'\""
END_MAIN_HEADINGS = (
    "references",
    "bibliography",
    "appendix",
    "appendices",
    "supplementarymaterial",
    "supplementalmaterial",
)


def arxiv_id_for_paper(paper):
    ext = paper.get("externalIds") or {}
    arxiv = extract_arxiv_id(ext.get("ArXiv"))
    if arxiv:
        return arxiv
    open_pdf = paper.get("openAccessPdf") or {}
    for value in [paper.get("url"), open_pdf.get("url"), ext.get("DOI")]:
        arxiv = extract_arxiv_id(value)
        if arxiv:
            return arxiv
    return ""


def get_json(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return {"_err": e.code}
        except Exception:
            time.sleep(3 * (attempt + 1))
    return {"_err": "retries"}


def get_bytes(url, timeout=60, max_bytes=None, headers=None):
    req = urllib.request.Request(url, headers=headers or UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # HTTPResponse needs an omitted size for an unbounded chunked read.
        data = r.read(max_bytes + 1) if max_bytes else r.read()
    if max_bytes and len(data) > max_bytes:
        raise ValueError(f"response exceeded {max_bytes} bytes: {url}")
    return data


def norm_repo(url):
    url = (url or "").strip().rstrip(TRAILING_URL_JUNK)
    m = re.search(
        r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", url, flags=re.I
    )
    if not m:
        return None
    owner, name = m.group(1).strip("."), m.group(2).strip(".")
    name = re.sub(r"\.git$", "", name)
    if owner.lower() in SKIP_OWNERS or name.lower() in {"", "."} or not name:
        return None
    return f"https://github.com/{owner}/{name}"


def find_repos(text):
    repos = []
    seen = set()
    pattern = r"(?:https?://)?(?:www\.)?github\.com/[^\s)\]\}\"'<>]+"
    for m in re.finditer(pattern, text or "", flags=re.I):
        repo = norm_repo(m.group(0))
        key = repo.lower() if repo else ""
        if repo and key not in seen:
            repos.append(repo)
            seen.add(key)
    return repos


def arxiv_pdf_url(arxiv_id):
    # Omitting a version suffix asks arXiv for the latest version.
    return f"https://arxiv.org/pdf/{extract_arxiv_id(arxiv_id)}"


def abstract_block(html):
    match = re.search(
        r"<blockquote\b[^>]*class=[\"'][^\"']*\babstract\b[^\"']*[\"'][^>]*>"
        r"(.*?)</blockquote>",
        html or "",
        flags=re.I | re.S,
    )
    return html_lib.unescape(match.group(1)) if match else ""


def arxiv_page_repos(arxiv_id):
    """Collect direct repository links from the arXiv abstract itself."""
    html = get_bytes(arxiv_abs_url(arxiv_id), timeout=30).decode("utf-8", "replace")
    if not abstract_block(html):
        raise ValueError(f"arXiv response has no abstract: {arxiv_abs_url(arxiv_id)}")
    return find_repos(abstract_block(html))


def is_end_of_main_paper_heading(line):
    compact = re.sub(r"[^a-z0-9]", "", (line or "").casefold())
    if not compact:
        return False
    return any(
        compact.startswith(heading)
        or (
            len(compact) > 1
            and compact[0].isalpha()
            and compact[1:].startswith(heading)
        )
        or re.match(rf"^\d+{heading}", compact)
        for heading in END_MAIN_HEADINGS
    )


def truncate_at_end_of_main_paper(text):
    lines = (text or "").splitlines()
    for index, line in enumerate(lines):
        if is_end_of_main_paper_heading(line):
            return "\n".join(lines[:index]), True
    return text or "", False


def pdf_main_text_from_bytes(data):
    try:
        import fitz  # PyMuPDF, optional
    except ImportError:
        fitz = None
    if fitz is not None:
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                parts = []
                for page in doc:
                    page_text, reached_end = truncate_at_end_of_main_paper(page.get_text())
                    parts.append(page_text)
                    if reached_end:
                        break
                    parts.extend(
                        link["uri"]
                        for link in page.get_links()
                        if link.get("uri")
                    )
                return "\n".join(parts)
        except Exception:
            pass

    executable = shutil.which("pdftotext")
    if not executable:
        raise RuntimeError("PDF extraction requires PyMuPDF or the pdftotext executable")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
        source.write(data)
        source.flush()
        result = subprocess.run(
            [executable, "-layout", source.name, "-"],
            capture_output=True,
            check=False,
            timeout=120,
        )
    if result.returncode:
        error = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"pdftotext failed: {error}")
    text = result.stdout.decode("utf-8", "replace")
    text, _ = truncate_at_end_of_main_paper(text)
    return text


def arxiv_pdf_repos(arxiv_id):
    """Collect direct repository links from the latest PDF's main paper."""
    data = get_bytes(arxiv_pdf_url(arxiv_id), timeout=90, max_bytes=PDF_MAX_BYTES)
    return find_repos(pdf_main_text_from_bytes(data))


def github_readme(repo):
    parsed = re.match(r"https://github\.com/([^/]+)/([^/]+)$", repo)
    if not parsed:
        return None, "invalid_repo"
    owner, name = parsed.groups()
    url = f"{GITHUB_API}/repos/{owner}/{name}/readme"
    headers = github_headers("sampling-study", accept="application/vnd.github.raw+json")
    while True:
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read(README_MAX_BYTES + 1)
                response_headers = dict(response.headers)
        except urllib.error.HTTPError as exc:
            response_headers = dict(exc.headers)
            remaining = response_headers.get("X-RateLimit-Remaining")
            if exc.code in {403, 429} and (exc.code == 429 or remaining == "0"):
                sleep_for_github_budget(response_headers)
                continue
            if exc.code == 404:
                return None, "missing_readme"
            raise RuntimeError(f"GitHub returned HTTP {exc.code} for {repo} README") from exc
        if len(body) > README_MAX_BYTES:
            return None, "readme_too_large"
        remaining = response_headers.get("X-RateLimit-Remaining")
        if remaining is not None and int(remaining) <= 1:
            sleep_for_github_budget(response_headers)
        return decode_readme_response(body), "ok"


def sleep_for_github_budget(headers):
    wait = rate_limit_reset_wait(headers)
    print(f"  GitHub budget exhausted; sleeping {wait}s", file=sys.stderr)
    time.sleep(wait)


def discover_repos_from_arxiv(arxiv_id, title):
    candidates = {}
    for source, repos in [
        ("arxiv_abs", arxiv_page_repos(arxiv_id)),
        ("arxiv_pdf", arxiv_pdf_repos(arxiv_id)),
    ]:
        for repo in repos:
            entry = candidates.setdefault(
                repo.lower(), {"repo_url": repo, "sources": []}
            )
            if source not in entry["sources"]:
                entry["sources"].append(source)

    evaluated = []
    accepted = []
    for entry in candidates.values():
        readme, readme_status = github_readme(entry["repo_url"])
        match = readme_names_paper(readme, title) if readme else False
        result = {
            "repo_url": entry["repo_url"],
            "repo_link_source": "+".join(entry["sources"]),
            "readme_status": readme_status,
            "readme_title_match": match,
        }
        evaluated.append(result)
        if match:
            accepted.append(result)
    return accepted, evaluated


def all_papers(venue, year):
    """All S2 papers for `venue`/`year`, ordered by publication date ascending."""
    papers, token = [], None
    while True:
        url = (
            f"{S2_BULK}?venue={urllib.parse.quote(venue)}&year={year}"
            f"&sort=publicationDate:asc&fields={FIELDS}"
        )
        if token:
            url += f"&token={urllib.parse.quote(token)}"
        data = get_json(url)
        if data.get("_err"):
            raise RuntimeError(f"S2 error while fetching {venue} {year}: {data['_err']}")
        batch = data.get("data") or []
        papers.extend(batch)
        token = data.get("token")
        print(f"  fetched {len(papers)} papers", flush=True)
        if not token:
            break
        time.sleep(1)
    papers.sort(key=paper_sort_key)
    return papers


def paper_sort_key(paper):
    # S2 often has undated proceedings records. Put those after dated records;
    # otherwise a resume run can spend a long prefix on records with no arXiv id.
    date = paper.get("publicationDate")
    return (date is None, date or "", paper.get("title") or "")


def paper_key(paper):
    return paper.get("paperId") or paper.get("title") or json.dumps(paper, sort_keys=True)


def load_state(state_path):
    if not state_path.exists():
        return None
    with open(state_path) as f:
        return json.load(f)


def progress_counts(processed):
    return Counter(row.get("status", "unknown") for row in processed.values())


def show_progress(done, total, counts):
    if total <= 0:
        return
    width = 32
    filled = min(width, int(width * done / total))
    bar = "#" * filled + "-" * (width - filled)
    pct = 100 * done / total
    line = (
        f"\rchecking GitHub links [{bar}] {done}/{total} "
        f"({pct:5.1f}%) kept={counts.get('kept', 0)} "
        f"no_arxiv={counts.get('no_arxiv_id', 0)} "
        f"no_repo={counts.get('no_repo_found', 0)} "
        f"fetch_error={counts.get('fetch_error', 0)} "
        f"no_title_match={counts.get('no_readme_title_match', 0)}"
    )
    sys.stderr.write(line)
    sys.stderr.flush()


def make_output_row(conf, venue, year, paper, arxiv_id, repo, repo_source):
    return {
        "conference": conf,
        "venue_raw": paper.get("venue") or venue,
        "year": paper.get("year") or year,
        "title": paper.get("title"),
        "arxiv_id": arxiv_id,
        "repo_url": repo,
        "paper_url_abs": arxiv_abs_url(arxiv_id),
        "publicationDate": paper.get("publicationDate"),
        "s2_paper_id": paper.get("paperId"),
        "repo_link_source": repo_source,
    }


def arxiv_coverage(papers):
    return sum(1 for paper in papers if arxiv_id_for_paper(paper))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--venue", default=os.environ.get("VENUE", "NeurIPS"))
    p.add_argument("--year", type=int, default=int(os.environ.get("YEAR", "2025")))
    p.add_argument("--conference", default=os.environ.get("CONFERENCE", None))
    p.add_argument("--out", default=os.environ.get("SAMPLE_OUT", None))
    p.add_argument("--state", default=os.environ.get("SAMPLE_STATE", None))
    p.add_argument("--refresh-paper-list", action="store_true")
    p.add_argument(
        "--retry-status", nargs="+", default=[],
        choices=["no_arxiv_id", "no_repo_found", "no_readme_title_match", "fetch_error"],
        help="recheck only these completed statuses, preserving all selected pairs",
    )
    p.add_argument("--sleep", type=float, default=float(os.environ.get("ARXIV_SLEEP", "0.34")))
    return p.parse_args()


def main():
    args = parse_args()
    conf = args.conference or args.venue
    slug = venue_slug(args.venue)
    out = BASE / (args.out or f"{slug}_{args.year}_buffer.json")
    state_path = BASE / (args.state or f"{slug}_{args.year}_buffer.state.json")

    state = load_state(state_path) or {}
    if (
        args.refresh_paper_list
        or state.get("venue") != args.venue
        or state.get("year") != args.year
        or not state.get("papers")
    ):
        print(f"{args.venue} {args.year}: fetching full S2 paper list", flush=True)
        papers = all_papers(args.venue, args.year)
        state = {
            "sample_version": SAMPLE_VERSION,
            "venue": args.venue,
            "conference": conf,
            "year": args.year,
            "papers": papers,
            "processed": {},
            "selected": [],
        }
        atomic_write_json(state_path, state)
    else:
        papers = sorted(state["papers"], key=paper_sort_key)
        if papers != state["papers"]:
            state["papers"] = papers
            atomic_write_json(state_path, state)
        print(
            f"{args.venue} {args.year}: resuming with {len(papers)} cached S2 papers",
            flush=True,
        )
        if state.get("sample_version") != SAMPLE_VERSION:
            print(
                "sampling rule changed; retaining the cached paper list and "
                "rechecking every paper",
                flush=True,
            )
            state["sample_version"] = SAMPLE_VERSION
            state["processed"] = {}
            state["selected"] = []
            atomic_write_json(state_path, state)
    print(
        f"S2 arXiv-id coverage: {arxiv_coverage(papers)}/{len(papers)} papers",
        file=sys.stderr,
        flush=True,
    )

    selected = state.setdefault("selected", [])
    processed = state.setdefault("processed", {})
    retry_keys = {
        key for key, record in processed.items()
        if record["status"] in args.retry_status
    }
    atomic_write_json(out, selected)

    kept_now = 0
    counts = progress_counts(processed)
    if not os.environ.get("GITHUB_TOKEN"):
        print(
            "warning: GITHUB_TOKEN is not set; README validation is limited to "
            "60 GitHub API requests per hour",
            file=sys.stderr,
        )
    print(f"Checking GitHub links for {len(papers)} papers...", file=sys.stderr, flush=True)
    show_progress(len(processed), len(papers), counts)
    for paper in papers:
        key = paper_key(paper)
        if key in processed and key not in retry_keys:
            continue

        arxiv_id = arxiv_id_for_paper(paper)
        record = {"title": paper.get("title"), "arxiv_id": arxiv_id}
        if not arxiv_id:
            record["status"] = "no_arxiv_id"
        else:
            try:
                accepted, evaluated = discover_repos_from_arxiv(
                    arxiv_id, paper.get("title") or ""
                )
            except (OSError, ValueError, RuntimeError) as exc:
                record.update(status="fetch_error", error=f"{type(exc).__name__}: {exc}")
                if key in processed:
                    counts[processed[key]["status"]] -= 1
                processed[key] = record
                counts["fetch_error"] += 1
                atomic_write_json(state_path, state)
                print(f"\n  {paper.get('title')}: {record['error']}", file=sys.stderr)
                time.sleep(args.sleep)
                show_progress(len(processed), len(papers), counts)
                continue
            time.sleep(args.sleep)
            if not evaluated:
                record["status"] = "no_repo_found"
            elif not accepted:
                record["status"] = "no_readme_title_match"
                record["repo_candidates"] = evaluated
            else:
                rows = [
                    make_output_row(
                        conf,
                        args.venue,
                        args.year,
                        paper,
                        arxiv_id,
                        candidate["repo_url"],
                        candidate["repo_link_source"],
                    )
                    for candidate in accepted
                ]
                selected.extend(rows)
                kept_now += len(rows)
                record["status"] = "kept"
                record["repo_urls"] = [c["repo_url"] for c in accepted]
                record["repo_candidates"] = evaluated

        if key in processed:
            counts[processed[key]["status"]] -= 1
        processed[key] = record
        counts[record["status"]] += 1
        if record["status"] == "kept":
            atomic_write_json(out, selected)
        atomic_write_json(state_path, state)
        show_progress(len(processed), len(papers), counts)

    atomic_write_json(out, selected)
    atomic_write_json(state_path, state)
    if papers:
        sys.stderr.write("\n")
        sys.stderr.flush()
    done = len(processed)
    print(
        f"DONE {args.venue} {args.year}: {len(selected)} paper-repo pairs kept "
        f"({kept_now} new pairs), {done}/{len(papers)} papers processed -> {out.name}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; rerun the same command to resume.", file=sys.stderr)
        raise
