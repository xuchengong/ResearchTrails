"""Minimal, dependency-free .env loader.

Loads KEY=VALUE lines from a `.env` file at the repository root into os.environ.
Existing environment variables are NOT overridden, so an explicit
`GITHUB_TOKEN=... python script.py` on the command line still takes precedence.

Usage (from any script in the repo):
    from envload import load_env
    load_env()
"""
import os
from pathlib import Path


def find_env_file(start=None):
    """Walk upward from `start` (default: this file's directory) for a `.env`."""
    here = Path(start or Path(__file__).resolve().parent)
    for d in [here, *here.parents]:
        candidate = d / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(path=None, override=False):
    """Load KEY=VALUE pairs from `.env` into os.environ. Returns loaded keys."""
    env_path = Path(path) if path else find_env_file()
    if not env_path or not env_path.is_file():
        return []

    loaded = []
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded
