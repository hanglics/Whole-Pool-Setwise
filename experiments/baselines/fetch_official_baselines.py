#!/usr/bin/env python3
"""Fetch or offline-verify the exact external baseline revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "upstreams.lock.json"
UPSTREAM_ROOT = ROOT / ".upstreams"


def run_git(arguments: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock() -> dict:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported upstream lock schema.")
    return lock


def split_checkout_status(status: str) -> tuple[list[str], list[str]]:
    """Separate meaningful source drift from untracked Python bytecode caches."""
    meaningful = []
    runtime_artifacts = []
    for line in status.splitlines():
        path = line[3:] if len(line) >= 4 else ""
        parts = PurePosixPath(path).parts
        if (
            line.startswith("?? ")
            and "__pycache__" in parts
            and PurePosixPath(path).suffix in {".pyc", ".pyo"}
        ):
            runtime_artifacts.append(path)
        else:
            meaningful.append(line)
    return meaningful, runtime_artifacts


def checkout_status(checkout: Path) -> tuple[str, list[str]]:
    raw = run_git(["status", "--porcelain", "--untracked-files=all"], checkout)
    meaningful, runtime_artifacts = split_checkout_status(raw)
    return "\n".join(meaningful), runtime_artifacts


def verify_one(name: str, spec: dict) -> dict:
    checkout = UPSTREAM_ROOT / name
    if not (checkout / ".git").is_dir():
        raise FileNotFoundError(f"Missing checkout: {checkout}")
    commit = run_git(["rev-parse", "HEAD"], checkout)
    if commit != spec["commit"]:
        raise ValueError(f"{name}: expected commit {spec['commit']}, found {commit}")
    dirty, runtime_artifacts = checkout_status(checkout)
    if dirty:
        raise ValueError(f"{name}: checkout is dirty:\n{dirty}")
    verified = {}
    for relative, expected_hash in spec["files"].items():
        path = checkout / relative
        if not path.is_file():
            raise FileNotFoundError(f"{name}: missing locked file {relative}")
        actual = file_sha256(path)
        if actual != expected_hash:
            raise ValueError(
                f"{name}: SHA-256 mismatch for {relative}: expected {expected_hash}, got {actual}"
            )
        verified[relative] = actual
    return {
        "name": name,
        "path": str(checkout),
        "commit": commit,
        "files": verified,
        "ignored_runtime_artifacts": runtime_artifacts,
    }


def fetch_one(name: str, spec: dict) -> None:
    checkout = UPSTREAM_ROOT / name
    if not checkout.exists():
        run_git(["clone", "--no-checkout", spec["repository"], str(checkout)])
    elif not (checkout / ".git").is_dir():
        raise ValueError(f"Refusing non-git upstream path: {checkout}")
    dirty, _runtime_artifacts = checkout_status(checkout)
    worktree_entries = [path for path in checkout.iterdir() if path.name != ".git"]
    # A fresh --no-checkout clone reports every tracked file as deleted even
    # though no worktree content exists yet. That is the sole dirty state safe
    # to materialize automatically.
    if dirty and worktree_entries:
        raise ValueError(f"Refusing to modify dirty checkout: {checkout}")
    # Fetch the pin explicitly; never resolve a branch or update to HEAD.
    run_git(["fetch", "origin", spec["commit"]], checkout)
    run_git(["checkout", "--detach", spec["commit"]], checkout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["all", "tourrank", "liu_fullrank"], default="all")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock = load_lock()
    names = list(lock["upstreams"]) if args.baseline == "all" else [args.baseline]
    UPSTREAM_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in names:
        spec = lock["upstreams"][name]
        if not args.verify_only:
            fetch_one(name, spec)
        reports.append(verify_one(name, spec))
    print(json.dumps({"verified": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
