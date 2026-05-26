#!/usr/bin/env python3
"""Pair WP-DE against other methods using per-query trec_eval rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


METHODS = (
    "topdown_bubblesort",
    "topdown_heapsort",
    "bottomup_bubblesort",
    "bottomup_heapsort",
    "maxcontext_topdown",
    "maxcontext_bottomup",
)
WPDE = "maxcontext_dualend"
DEFAULT_MODELS = (
    "qwen3-5-0-8b",
    "qwen3-5-2b",
    "qwen3-5-4b",
    "qwen3-5-9b",
    "qwen3-5-27b",
    "meta-llama-3-1-8b-instruct",
    "ministral-3-3b-instruct-2512",
    "ministral-3-8b-instruct-2512",
    "ministral-3-14b-instruct-2512",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/emnlp/main/phase_b1_dl"))
    parser.add_argument("--output", type=Path, default=Path("results/emnlp/analysis/wpde_significance.csv"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--datasets", nargs="+", default=["dl19", "dl20"])
    parser.add_argument("--pools", nargs="+", type=int, default=[10, 20, 30, 40, 50, 100])
    parser.add_argument("--metrics", nargs="+", default=["ndcg_cut_10"])
    parser.add_argument("--comparators", nargs="+", default=list(METHODS))
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=929)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def read_eval(path: Path, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.exists():
        return values
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 3 and parts[0] == metric and parts[1] != "all":
                values[parts[1]] = float(parts[2])
    return values


def paired_randomization(a: np.ndarray, b: np.ndarray, samples: int, seed: int) -> float:
    diff = a - b
    observed = abs(float(diff.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, diff.size))
    permuted = np.abs((signs * diff).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed - 1e-12) + 1.0) / (samples + 1.0))


def paired_arrays(a: dict[str, float], b: dict[str, float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    qids = sorted(set(a) & set(b))
    return qids, np.array([a[qid] for qid in qids]), np.array([b[qid] for qid in qids])


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    test_index = 0

    for model in args.models:
        for dataset in args.datasets:
            for pool in args.pools:
                pool_tag = f"pool{pool:02d}"
                for metric in args.metrics:
                    wpde_eval = args.root / model / dataset / WPDE / pool_tag / f"{WPDE}.eval"
                    wpde_values = read_eval(wpde_eval, metric)
                    if not wpde_values:
                        continue
                    for method in args.comparators:
                        method_eval = args.root / model / dataset / method / pool_tag / f"{method}.eval"
                        method_values = read_eval(method_eval, metric)
                        if not method_values:
                            continue
                        qids, wpde_arr, method_arr = paired_arrays(wpde_values, method_values)
                        if not qids:
                            continue
                        p_raw = paired_randomization(
                            wpde_arr,
                            method_arr,
                            samples=args.samples,
                            seed=args.seed + test_index,
                        )
                        rows.append(
                            {
                                "model": model,
                                "dataset": dataset,
                                "pool": pool,
                                "metric": metric,
                                "wpde_method": WPDE,
                                "comparator": method,
                                "n_queries": len(qids),
                                "wpde_mean": float(wpde_arr.mean()),
                                "comparator_mean": float(method_arr.mean()),
                                "delta_wpde_minus_comparator": float(wpde_arr.mean() - method_arr.mean()),
                                "p_raw": p_raw,
                            }
                        )
                        test_index += 1

    m = len(rows)
    for row in rows:
        p_adj = min(1.0, float(row["p_raw"]) * m) if m else 1.0
        row["p_bonferroni"] = p_adj
        row["significant"] = p_adj < args.alpha
        delta = float(row["delta_wpde_minus_comparator"])
        row["direction"] = "wpde_higher" if delta > 0 else "wpde_lower" if delta < 0 else "tie"

    fieldnames = [
        "model",
        "dataset",
        "pool",
        "metric",
        "wpde_method",
        "comparator",
        "n_queries",
        "wpde_mean",
        "comparator_mean",
        "delta_wpde_minus_comparator",
        "p_raw",
        "p_bonferroni",
        "significant",
        "direction",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} WP-DE comparisons to {args.output}")


if __name__ == "__main__":
    main()
