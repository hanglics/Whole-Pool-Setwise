#!/usr/bin/env python3
"""Aggregate EMNLP MaxContext position-bias logs by family and pool.

This script intentionally analyzes only MaxContext comparison logs under the
EMNLP main tree. Standard TopDown/BottomUp logs are produced by the EMNLP
launchers but are outside this paper track.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

from position_bias import (
    load_comparison_entries,
    render_position_bias_summary,
    summarize_position_bias,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None


DEFAULT_MAIN_ROOT = Path("results/emnlp/main")
DEFAULT_OUTPUT_ROOT = Path("results/emnlp/analysis/position_bias_emnlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EMNLP MaxContext position-bias aggregation by model family."
    )
    parser.add_argument(
        "--main-root",
        type=Path,
        default=DEFAULT_MAIN_ROOT,
        help="Root containing results/emnlp/main/{tag}/... outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for grouped CSV, text, and plot outputs.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional main-matrix tag to restrict discovery.",
    )
    return parser.parse_args()


def model_family_from_tag(model_tag: str) -> str:
    normalized = model_tag.lower()
    if normalized.startswith("qwen3-5-"):
        return "qwen3.5"
    if normalized.startswith("meta-llama-"):
        return "llama3.1"
    if normalized.startswith("ministral-3-"):
        return "mistral3"
    if normalized.startswith("qwen3-") and not normalized.startswith("qwen3-5-"):
        return "qwen3"
    return "unknown"


def discover_logs(main_root: Path, tag: str | None = None) -> list[Path]:
    root = main_root / tag if tag else main_root
    pattern = "*/*/maxcontext_*/pool*/maxcontext_*_comparisons.jsonl" if tag else "*/*/*/maxcontext_*/pool*/maxcontext_*_comparisons.jsonl"
    return sorted(root.glob(pattern), key=lambda path: path.as_posix())


def parse_log_path(main_root: Path, log_path: Path) -> dict[str, str | int]:
    rel = log_path.relative_to(main_root)
    if len(rel.parts) != 6:
        raise ValueError(f"Unexpected EMNLP log path shape: {log_path}")
    tag, model_tag, dataset_tag, method, pool_tag, _ = rel.parts
    if not method.startswith("maxcontext_"):
        raise ValueError(f"Non-MaxContext log passed to EMNLP position-bias analysis: {log_path}")
    match = re.fullmatch(r"pool(\d+)(?:_(reverse|shuffle))?", pool_tag)
    if not match:
        raise ValueError(f"Unexpected pool tag in {log_path}")
    return {
        "tag": tag,
        "model_tag": model_tag,
        "model_family": model_family_from_tag(model_tag),
        "dataset_tag": dataset_tag,
        "method": method,
        "pool_size": int(match.group(1)),
        "condition": match.group(2) or "forward",
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_plot(path: Path, rows: list[dict[str, object]], title: str) -> bool:
    if plt is None or not rows:
        return False
    x = [int(row["position_index"]) + 1 for row in rows]
    y = [float(row["frequency"]) for row in rows]
    expected = [float(row["expected"]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, y, color="#4c78a8", label="observed")
    ax.plot(x, expected, color="#d62728", linewidth=1.5, label="uniform")
    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Selection frequency")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def read_eval_metric(eval_path: Path, metric: str = "ndcg_cut_10") -> dict[str, float]:
    values: dict[str, float] = {}
    if not eval_path.exists():
        return values
    with eval_path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0] == metric:
                values[parts[1]] = float(parts[2])
    return values


def sign_test_p_value(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n == 0:
        return None
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def build_condition_delta_rows(main_root: Path, logs: list[Path]) -> list[dict[str, object]]:
    evals_by_cell: dict[
        tuple[str, str, str, int, str],
        tuple[dict[str, str | int], dict[str, float]],
    ] = {}
    for log_path in logs:
        meta = parse_log_path(main_root, log_path)
        metrics = read_eval_metric(log_path.parent / f"{meta['method']}.eval")
        if not metrics:
            continue
        key = (
            str(meta["model_tag"]),
            str(meta["dataset_tag"]),
            str(meta["method"]),
            int(meta["pool_size"]),
            str(meta["condition"]),
        )
        evals_by_cell.setdefault(key, (meta, metrics))

    by_cell: dict[
        tuple[str, str, str, int],
        dict[str, tuple[dict[str, str | int], dict[str, float]]],
    ] = defaultdict(dict)
    for key, value in evals_by_cell.items():
        model_tag, dataset_tag, method, pool_size, condition = key
        by_cell[(model_tag, dataset_tag, method, pool_size)][condition] = value

    rows: list[dict[str, object]] = []
    for cell_key, condition_map in sorted(by_cell.items()):
        if "forward" not in condition_map:
            continue
        _, forward_values = condition_map["forward"]
        for condition in ("reverse", "shuffle"):
            if condition not in condition_map:
                continue
            meta, condition_values = condition_map[condition]
            qids = sorted(
                qid for qid in set(forward_values) & set(condition_values) if qid != "all"
            )
            deltas = [condition_values[qid] - forward_values[qid] for qid in qids]
            positives = sum(delta > 0 for delta in deltas)
            negatives = sum(delta < 0 for delta in deltas)
            all_delta = None
            if "all" in forward_values and "all" in condition_values:
                all_delta = condition_values["all"] - forward_values["all"]
            rows.append(
                {
                    "model_tag": cell_key[0],
                    "model_family": model_family_from_tag(cell_key[0]),
                    "dataset_tag": cell_key[1],
                    "method": cell_key[2],
                    "pool_size": cell_key[3],
                    "condition": condition,
                    "forward_tag": condition_map["forward"][0]["tag"],
                    "condition_tag": meta["tag"],
                    "metric": "ndcg_cut_10",
                    "aggregate_delta": all_delta,
                    "paired_qids": len(qids),
                    "mean_query_delta": sum(deltas) / len(deltas) if deltas else None,
                    "positive_qids": positives,
                    "negative_qids": negatives,
                    "tied_qids": len(deltas) - positives - negatives,
                    "sign_test_p": sign_test_p_value(positives, negatives),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    logs = discover_logs(args.main_root, args.tag)
    if not logs:
        raise SystemExit(f"No MaxContext comparison logs found under {args.main_root}")

    grouped: dict[tuple[str, str, str, int, str, str], list[dict[str, object]]] = defaultdict(list)
    source_counts: dict[tuple[str, str, str, int, str, str], int] = defaultdict(int)
    for log_path in logs:
        meta = parse_log_path(args.main_root, log_path)
        entries = load_comparison_entries([log_path])
        by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
        for entry in entries:
            by_type[str(entry["type"])].append(entry)
        for comparison_type, type_entries in by_type.items():
            key = (
                str(meta["model_family"]),
                str(meta["dataset_tag"]),
                str(meta["method"]),
                int(meta["pool_size"]),
                str(meta["condition"]),
                comparison_type,
            )
            grouped[key].extend(type_entries)
            source_counts[key] += 1

    index_rows = []
    for key, entries in sorted(grouped.items()):
        family, dataset_tag, method, pool_size, condition, comparison_type = key
        summary = summarize_position_bias(entries)
        if summary is None:
            continue
        type_summary = summary["types"][comparison_type]
        out_dir = (
            args.output_root
            / family
            / dataset_tag
            / method
            / f"pool{pool_size}"
            / condition
            / comparison_type
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        lines = render_position_bias_summary(summary)
        (out_dir / "position_bias.txt").write_text("\n".join(lines) + "\n")

        metadata = {
            "model_family": family,
            "dataset_tag": dataset_tag,
            "method": method,
            "pool_size": pool_size,
            "condition": condition,
            "comparison_type": comparison_type,
        }
        selection_rows = [{**metadata, **row} for row in type_summary["selection_rows"]]
        write_csv(
            out_dir / "position_bias.csv",
            selection_rows,
            [
                "model_family",
                "dataset_tag",
                "method",
                "pool_size",
                "condition",
                "comparison_type",
                "position_index",
                "label",
                "count",
                "frequency",
                "expected",
                "bias",
            ],
        )

        accuracy_rows = [{**metadata, **row} for row in type_summary["accuracy_rows"]]
        write_csv(
            out_dir / "position_accuracy.csv",
            accuracy_rows,
            [
                "model_family",
                "dataset_tag",
                "method",
                "pool_size",
                "condition",
                "comparison_type",
                "position_index",
                "label",
                "correct",
                "total",
                "accuracy",
            ],
        )

        write_csv(
            out_dir / "summary.csv",
            [
                {
                    **metadata,
                    "n_comparisons": type_summary["n_comparisons"],
                    "valid_selections": type_summary["valid_selections"],
                    "n_positions": type_summary["n_positions"],
                    "chi2": type_summary["chi2"],
                    "source_logs": source_counts[key],
                }
            ],
            [
                "model_family",
                "dataset_tag",
                "method",
                "pool_size",
                "condition",
                "comparison_type",
                "n_comparisons",
                "valid_selections",
                "n_positions",
                "chi2",
                "source_logs",
            ],
        )

        plotted = write_plot(
            out_dir / "position_bias.png",
            type_summary["selection_rows"],
            f"{family} {dataset_tag} {method} pool{pool_size} {condition} {comparison_type}",
        )
        if not plotted:
            (out_dir / "PLOT_SKIPPED.txt").write_text("matplotlib is not available or no rows were present.\n")

        index_rows.append(
            {
                **metadata,
                "n_comparisons": type_summary["n_comparisons"],
                "valid_selections": type_summary["valid_selections"],
                "source_logs": source_counts[key],
                "output_dir": out_dir.as_posix(),
            }
        )

    write_csv(
        args.output_root / "index.csv",
        index_rows,
        [
            "model_family",
            "dataset_tag",
            "method",
            "pool_size",
            "condition",
            "comparison_type",
            "n_comparisons",
            "valid_selections",
            "source_logs",
            "output_dir",
        ],
    )
    delta_rows = build_condition_delta_rows(args.main_root, logs)
    write_csv(
        args.output_root / "condition_deltas.csv",
        delta_rows,
        [
            "model_tag",
            "model_family",
            "dataset_tag",
            "method",
            "pool_size",
            "condition",
            "forward_tag",
            "condition_tag",
            "metric",
            "aggregate_delta",
            "paired_qids",
            "mean_query_delta",
            "positive_qids",
            "negative_qids",
            "tied_qids",
            "sign_test_p",
        ],
    )
    if delta_rows:
        lines = [
            "# Phase F Position-Bias Deltas",
            "",
            "| Model | Dataset | Method | Pool | Condition | Agg Delta nDCG@10 | Mean Query Delta | qids | sign p |",
            "|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
        for row in delta_rows:
            aggregate_delta = row["aggregate_delta"]
            mean_query_delta = row["mean_query_delta"]
            sign_p = row["sign_test_p"]
            lines.append(
                "| {model} | {dataset} | {method} | {pool} | {condition} | {agg} | {mean_delta} | {qids} | {p} |".format(
                    model=row["model_tag"],
                    dataset=row["dataset_tag"],
                    method=row["method"],
                    pool=row["pool_size"],
                    condition=row["condition"],
                    agg="" if aggregate_delta is None else f"{float(aggregate_delta):.6f}",
                    mean_delta="" if mean_query_delta is None else f"{float(mean_query_delta):.6f}",
                    qids=row["paired_qids"],
                    p="" if sign_p is None else f"{float(sign_p):.6f}",
                )
            )
        (args.output_root / "condition_deltas.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(index_rows)} grouped position-bias outputs to {args.output_root}")


if __name__ == "__main__":
    main()
