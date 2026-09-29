#!/usr/bin/env python3
"""Capture the immutable code/runtime inputs used by paper experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    'requirements.txt',
    'run.py',
    'llmrankers/rankers.py',
    'llmrankers/_processor_adapter.py',
    'llmrankers/setwise.py',
    'llmrankers/setwise_extended.py',
    'llmrankers/experiment_controls.py',
    'experiments/baselines/activate_conda.sh',
    'experiments/baselines/capture_provenance.py',
    'experiments/baselines/collect_results.py',
    'experiments/baselines/evaluate.sh',
    'experiments/baselines/fetch_official_baselines.py',
    'experiments/baselines/finalize_native_attempt.py',
    'experiments/baselines/finalize_official_attempt.py',
    'experiments/baselines/model_revisions.lock.json',
    'experiments/baselines/official_baselines.py',
    'experiments/baselines/open_model_backend.py',
    'experiments/baselines/run_liu.sh',
    'experiments/baselines/run_liu_zero_shot_model_matched.py',
    'experiments/baselines/run_setwise.sh',
    'experiments/baselines/run_tourrank.sh',
    'experiments/baselines/run_tourrank_model_matched.py',
    'experiments/baselines/submit_jobs.sh',
    'experiments/baselines/upstreams.lock.json',
    'experiments/baselines/environment.sh',
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot() -> dict:
    files = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Declared experiment source is missing: {path}")
        files[relative] = sha256_file(path)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "files": files,
    }


def runtime_snapshot() -> dict:
    packages = {}
    for name in (
        "torch", "transformers", "accelerate", "ir-datasets", "pyserini",
        "ftfy", "dacite", "vllm",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu = []
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_commit = None
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True
        )
    except subprocess.CalledProcessError:
        git_status = ""
    return {
        "source_snapshot": source_snapshot(),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status_sha256": hashlib.sha256(git_status.encode("utf-8")).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpus": gpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", action="store_true")
    parser.add_argument("--runtime-json", action="store_true")
    args = parser.parse_args()
    if args.source_sha == args.runtime_json:
        parser.error("choose exactly one of --source-sha or --runtime-json")
    if args.source_sha:
        print(source_snapshot()["sha256"])
    else:
        print(json.dumps(runtime_snapshot(), sort_keys=True))


if __name__ == "__main__":
    main()
