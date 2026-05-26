"""
Repeated-run stability analysis for LLM ranking experiments.

The script expects a results directory with duplicate runs such as:

    results/maxcontext_dualend/qwen3-4b-dl19/stability-test-runs/test_run_v1
    results/maxcontext_dualend/qwen3-4b-dl19/stability-test-runs/test_run_v2
    ...

It reads every `.eval` file under the duplicate run directories and computes
stability statistics across runs for each method/top-k/metric combination. It
also parses `.log` files for cost and fallback metrics.

Outputs:
    - evaluation_stability.csv: aggregate `all` eval rows across runs
    - per_query_stability.csv: per-query eval rows across runs
    - per_query_stability_summary.csv: query-level std/range summary
    - log_stability.csv: log metric stability across runs
    - STABILITY_REPORT.md: compact Markdown summary

Usage:
    python3 analysis/repeated_run_stability.py \
        --results-root results/maxcontext_dualend/qwen3-4b-dl19/stability-test-runs
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


EVAL_SUFFIX = ".eval"
LOG_SUFFIX = ".log"
DEFAULT_PRIMARY_METRIC = "ndcg_cut_10"
DEFAULT_RESULTS_ROOT = Path("results/maxcontext_dualend/qwen3-4b-dl19/stability-test-runs")
STABILITY_RUNS_DIRNAME = "stability-test-runs"
STABILITY_ANALYSIS_DIRNAME = "stability-analysis"

LOG_PATTERNS = {
    "avg_comparisons": r"Avg comparisons:\s*([0-9.eE+-]+)",
    "avg_prompt_tokens": r"Avg prompt tokens:\s*([0-9.eE+-]+)",
    "avg_completion_tokens": r"Avg completion tokens:\s*([0-9.eE+-]+)",
    "avg_time_per_query": r"Avg time per query:\s*([0-9.eE+-]+)",
    "avg_parse_fallbacks": r"Avg parse fallbacks:\s*([0-9.eE+-]+)",
    "avg_lexical_refusal_fallbacks": r"Avg lexical refusal fallbacks:\s*([0-9.eE+-]+)",
    "avg_numeric_out_of_range_fallbacks": r"Avg numeric out-of-range fallbacks:\s*([0-9.eE+-]+)",
}

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Directory containing duplicate test_run_v* result folders. "
            "If a model root with a stability-test-runs child is provided, "
            "that child is used automatically."
        ),
    )
    parser.add_argument(
        "--run-glob",
        default="test_run_v*",
        help="Glob for duplicate run directories under results root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to the sibling stability-analysis "
            "directory when results root is stability-test-runs, otherwise "
            "<results-root>/stability-analysis."
        ),
    )
    parser.add_argument(
        "--primary-metric",
        default=DEFAULT_PRIMARY_METRIC,
        help="Metric to highlight in the Markdown report.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=10,
        help="Expected duplicate run count. Used only for warnings/reporting.",
    )
    parser.add_argument(
        "--skip-per-query",
        action="store_true",
        help="Only write aggregate eval stability and log stability.",
    )
    return parser.parse_args()


def run_sort_key(run_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", run_id)
    if match:
        return int(match.group(1)), run_id
    return math.inf, run_id


def topk_sort_key(topk: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", topk)
    if match:
        return int(match.group(1)), topk
    return math.inf, topk


def metric_sort_key(metric: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.+?)_(\d+)", metric)
    if match:
        return match.group(1), int(match.group(2)), metric
    return metric, math.inf, metric


def infer_metadata(path: Path, results_root: Path) -> dict[str, str]:
    rel = path.relative_to(results_root)
    parts = rel.parts
    run_id = parts[0]

    topk = "unknown"
    condition = "forward"
    for part in parts:
        match = re.fullmatch(r"(top\d+)(?:_(reverse|shuffle))?", part)
        if match:
            topk = match.group(1)
            condition = match.group(2) or "forward"
            break
    model = next((part for part in parts if re.fullmatch(r".+-dl\d+", part)), "unknown")
    stem = path.stem

    rel_text = rel.as_posix()
    if "/phase1/" in f"/{rel_text}/" and stem == "maxcontext_dualend":
        method = "dualend"
        method_family = "phase1"
    elif "/max-context/topdown/" in f"/{rel_text}/":
        method = "maxctx_topdown"
        method_family = "max-context"
    elif "/max-context/bottomup/" in f"/{rel_text}/":
        method = "maxctx_bottomup"
        method_family = "max-context"
    elif "/original/bottomup/ws-3/" in f"/{rel_text}/":
        method = f"orig_bu_ws3_{stem.replace('bottomup_', '')}"
        method_family = "original_bottomup_ws3"
    elif "/original/bottomup/ws-ps/" in f"/{rel_text}/":
        method = f"orig_bu_wsps_{stem.replace('bottomup_', '')}"
        method_family = "original_bottomup_wsps"
    elif "/original/bottomup/" in f"/{rel_text}/":
        method = f"orig_bu_{stem.replace('bottomup_', '')}"
        method_family = "original_bottomup"
    elif "/original/ws-3/" in f"/{rel_text}/":
        method = f"orig_ws3_{stem.replace('topdown_', '')}"
        method_family = "original_ws3"
    elif "/original/ws-ps/" in f"/{rel_text}/":
        method = f"orig_wsps_{stem.replace('topdown_', '')}"
        method_family = "original_wsps"
    else:
        method = stem
        method_family = "unknown"

    return {
        "run_id": run_id,
        "model": model,
        "topk": topk,
        "condition": condition,
        "method": method,
        "method_family": method_family,
        "source_path": str(path),
    }


def parse_eval_file(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            metric, qid, value = parts
            rows.append({
                "metric": metric,
                "qid": qid,
                "value": float(value),
            })
    return rows


def parse_log_file(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    parsed: dict[str, float] = {}
    for metric, pattern in LOG_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            parsed[metric] = float(match.group(1))
    if "avg_prompt_tokens" in parsed and "avg_completion_tokens" in parsed:
        parsed["avg_total_tokens"] = parsed["avg_prompt_tokens"] + parsed["avg_completion_tokens"]
    return parsed


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    midpoint = n // 2
    if n % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sample_variance(values: list[float], avg: float) -> float | None:
    if len(values) < 2:
        return None
    return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


def population_variance(values: list[float], avg: float) -> float:
    return sum((value - avg) ** 2 for value in values) / len(values)


def t_critical_95(df: int) -> float:
    if df <= 0:
        return math.nan
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    return 1.96


def summarize_values(values_by_run: dict[str, float]) -> dict[str, object]:
    run_ids = sorted(values_by_run, key=run_sort_key)
    values = [values_by_run[run_id] for run_id in run_ids]
    avg = mean(values)
    pop_var = population_variance(values, avg)
    pop_std = math.sqrt(pop_var)
    samp_var = sample_variance(values, avg)
    samp_std = math.sqrt(samp_var) if samp_var is not None else None
    stderr = (samp_std / math.sqrt(len(values))) if samp_std is not None else None
    ci_half_width = (
        t_critical_95(len(values) - 1) * stderr
        if stderr is not None
        else None
    )
    min_value = min(values)
    max_value = max(values)
    value_range = max_value - min_value
    coefficient_of_variation = (
        samp_std / abs(avg)
        if samp_std is not None and avg != 0.0
        else None
    )

    return {
        "n_runs": len(values),
        "run_ids": ";".join(run_ids),
        "mean": avg,
        "sample_std": samp_std,
        "sample_variance": samp_var,
        "population_std": pop_std,
        "population_variance": pop_var,
        "std_error": stderr,
        "ci95_low": avg - ci_half_width if ci_half_width is not None else None,
        "ci95_high": avg + ci_half_width if ci_half_width is not None else None,
        "min": min_value,
        "max": max_value,
        "range": value_range,
        "coefficient_of_variation": coefficient_of_variation,
    }


def format_number(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_number(row.get(field)) for field in fieldnames})


def collect_eval_values(
    eval_paths: Iterable[Path],
    results_root: Path,
) -> tuple[dict[tuple[str, str, str, str, str, str], dict[str, float]], dict[tuple[str, str, str, str, str, str, str], dict[str, float]]]:
    aggregate_groups: dict[tuple[str, str, str, str, str, str], dict[str, float]] = defaultdict(dict)
    per_query_groups: dict[tuple[str, str, str, str, str, str, str], dict[str, float]] = defaultdict(dict)

    for path in eval_paths:
        metadata = infer_metadata(path, results_root)
        for row in parse_eval_file(path):
            metric = str(row["metric"])
            qid = str(row["qid"])
            value = float(row["value"])

            aggregate_key = (
                metadata["model"],
                metadata["method_family"],
                metadata["method"],
                metadata["topk"],
                metadata["condition"],
                metric,
            )
            query_key = aggregate_key + (qid,)

            if qid == "all":
                aggregate_groups[aggregate_key][metadata["run_id"]] = value
            else:
                per_query_groups[query_key][metadata["run_id"]] = value

    return aggregate_groups, per_query_groups


def collect_log_values(
    log_paths: Iterable[Path],
    results_root: Path,
) -> dict[tuple[str, str, str, str, str, str], dict[str, float]]:
    log_groups: dict[tuple[str, str, str, str, str, str], dict[str, float]] = defaultdict(dict)
    for path in log_paths:
        metadata = infer_metadata(path, results_root)
        for metric, value in parse_log_file(path).items():
            key = (
                metadata["model"],
                metadata["method_family"],
                metadata["method"],
                metadata["topk"],
                metadata["condition"],
                metric,
            )
            log_groups[key][metadata["run_id"]] = value
    return log_groups


def group_to_rows(
    groups: dict[tuple[str, ...], dict[str, float]],
    key_fields: list[str],
    expected_run_ids: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, values_by_run in sorted(groups.items(), key=lambda item: row_sort_key(item[0])):
        row = dict(zip(key_fields, key))
        row.update(summarize_values(values_by_run))
        missing_runs = [run_id for run_id in expected_run_ids if run_id not in values_by_run]
        row["missing_run_ids"] = ";".join(missing_runs)
        for run_id in expected_run_ids:
            row[run_id] = values_by_run.get(run_id)
        rows.append(row)
    return rows


def row_sort_key(key: tuple[str, ...]) -> tuple[object, ...]:
    sortable = []
    for value in key:
        if re.fullmatch(r"top\d+", value):
            sortable.append(topk_sort_key(value))
        elif re.fullmatch(r".+?_\d+", value):
            sortable.append(metric_sort_key(value))
        else:
            sortable.append(value)
    return tuple(sortable)


def build_per_query_summary(
    per_query_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_query_rows:
        key = (
            str(row["model"]),
            str(row["method_family"]),
            str(row["method"]),
            str(row["topk"]),
            str(row["condition"]),
            str(row["metric"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: row_sort_key(item[0])):
        stds = [float(row["sample_std"]) for row in rows if row.get("sample_std") not in {None, ""}]
        ranges = [float(row["range"]) for row in rows if row.get("range") not in {None, ""}]
        if not stds:
            continue
        summary = dict(zip(["model", "method_family", "method", "topk", "condition", "metric"], key))
        summary.update({
            "n_queries": len(stds),
            "mean_query_std": mean(stds),
            "median_query_std": median(stds),
            "p90_query_std": quantile(stds, 0.90),
            "max_query_std": max(stds),
            "mean_query_range": mean(ranges),
            "median_query_range": median(ranges),
            "p90_query_range": quantile(ranges, 0.90),
            "max_query_range": max(ranges),
            "queries_std_gt_0_001": sum(value > 0.001 for value in stds),
            "queries_std_gt_0_005": sum(value > 0.005 for value in stds),
            "queries_std_gt_0_01": sum(value > 0.01 for value in stds),
        })
        summary_rows.append(summary)
    return summary_rows


def format_report_value(row: dict[str, object]) -> str:
    mean_value = row.get("mean")
    std_value = row.get("sample_std")
    cv_value = row.get("coefficient_of_variation")
    if not isinstance(mean_value, float):
        return "-"
    if isinstance(std_value, float):
        cv = f", cv={100.0 * cv_value:.2f}%" if isinstance(cv_value, float) else ""
        return f"{mean_value:.4f} +/- {std_value:.4f}{cv}"
    return f"{mean_value:.4f}"


def build_primary_metric_table(
    aggregate_rows: list[dict[str, object]],
    primary_metric: str,
) -> list[str]:
    primary_rows = [row for row in aggregate_rows if row["metric"] == primary_metric]
    methods = sorted({(str(row["method"]), str(row.get("condition", "forward"))) for row in primary_rows})
    topks = sorted({str(row["topk"]) for row in primary_rows}, key=topk_sort_key)
    by_key = {
        (str(row["method"]), str(row.get("condition", "forward")), str(row["topk"])): row
        for row in primary_rows
    }

    lines = [
        f"## Primary Metric: {primary_metric}",
        "",
        "| Method | Condition | " + " | ".join(topks) + " |",
        "|---|---" + "|---:" * len(topks) + "|",
    ]
    for method, condition in methods:
        cells = [format_report_value(by_key.get((method, condition, topk), {})) for topk in topks]
        lines.append(f"| `{method}` | {condition} | " + " | ".join(cells) + " |")
    return lines


def build_most_stable_table(
    aggregate_rows: list[dict[str, object]],
    primary_metric: str,
    limit: int = 10,
) -> list[str]:
    primary_rows = [
        row for row in aggregate_rows
        if row["metric"] == primary_metric and isinstance(row.get("sample_std"), float)
    ]
    primary_rows.sort(key=lambda row: (
        float(row["sample_std"]),
        str(row["method"]),
        str(row.get("condition", "forward")),
        str(row["topk"]),
    ))

    lines = [
        f"## Lowest Run-to-Run Std: {primary_metric}",
        "",
        "| Method | Condition | TopK | Mean | Sample Std | CV | Range |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows[:limit]:
        cv = row.get("coefficient_of_variation")
        lines.append(
            "| `{method}` | {condition} | {topk} | {mean:.4f} | {std:.6f} | {cv} | {range_value:.6f} |".format(
                method=row["method"],
                condition=row.get("condition", "forward"),
                topk=str(row["topk"]).replace("top", ""),
                mean=float(row["mean"]),
                std=float(row["sample_std"]),
                cv=f"{100.0 * cv:.3f}%" if isinstance(cv, float) else "-",
                range_value=float(row["range"]),
            )
        )
    return lines


def build_report(
    results_root: Path,
    run_ids: list[str],
    aggregate_rows: list[dict[str, object]],
    per_query_summary_rows: list[dict[str, object]],
    log_rows: list[dict[str, object]],
    primary_metric: str,
    expected_runs: int,
) -> str:
    missing_groups = [row for row in aggregate_rows if row.get("missing_run_ids")]
    lines = [
        "# Repeated-Run Stability Report",
        "",
        f"- Results root: `{results_root}`",
        f"- Duplicate runs discovered: {len(run_ids)} ({', '.join(run_ids)})",
        f"- Expected duplicate runs: {expected_runs}",
        f"- Aggregate eval groups: {len(aggregate_rows)}",
        f"- Log metric groups: {len(log_rows)}",
        f"- Aggregate groups with missing runs: {len(missing_groups)}",
        "",
    ]

    lines.extend(build_primary_metric_table(aggregate_rows, primary_metric))
    lines.append("")
    lines.extend(build_most_stable_table(aggregate_rows, primary_metric))

    if per_query_summary_rows:
        rows = [
            row for row in per_query_summary_rows
            if row["metric"] == primary_metric
        ]
        rows.sort(key=lambda row: (
            str(row["method"]),
            str(row.get("condition", "forward")),
            topk_sort_key(str(row["topk"])),
        ))
        lines.extend([
            "",
            f"## Query-Level Stability Summary: {primary_metric}",
            "",
            "| Method | Condition | TopK | Mean Query Std | P90 Query Std | Max Query Std | Queries Std > 0.01 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                "| `{method}` | {condition} | {topk} | {mean_std:.6f} | {p90_std:.6f} | {max_std:.6f} | {gt} |".format(
                    method=row["method"],
                    condition=row.get("condition", "forward"),
                    topk=str(row["topk"]).replace("top", ""),
                    mean_std=float(row["mean_query_std"]),
                    p90_std=float(row["p90_query_std"]),
                    max_std=float(row["max_query_std"]),
                    gt=row["queries_std_gt_0_01"],
                )
            )

    if log_rows:
        lines.extend([
            "",
            "## Log Stability Files",
            "",
            "Log metrics include comparisons, prompt/completion/total tokens, time per query, and fallback counts where present.",
        ])

    lines.extend([
        "",
        "## Output Files",
        "",
        "- `evaluation_stability.csv`: aggregate metric mean/std/variance across duplicate runs.",
    ])
    if per_query_summary_rows:
        lines.extend([
            "- `per_query_stability.csv`: per-query metric mean/std/variance across duplicate runs.",
            "- `per_query_stability_summary.csv`: compact query-level stability distribution.",
        ])
    lines.append("- `log_stability.csv`: log metric mean/std/variance across duplicate runs.")
    return "\n".join(lines) + "\n"


def discover_run_dirs(results_root: Path, run_glob: str) -> list[Path]:
    run_dirs = [path for path in results_root.glob(run_glob) if path.is_dir()]
    return sorted(run_dirs, key=lambda path: run_sort_key(path.name))


def resolve_results_root(results_root: Path) -> Path:
    stability_runs = results_root / STABILITY_RUNS_DIRNAME
    if stability_runs.is_dir():
        return stability_runs
    return results_root


def default_output_dir(results_root: Path) -> Path:
    if results_root.name == STABILITY_RUNS_DIRNAME:
        return results_root.parent / STABILITY_ANALYSIS_DIRNAME
    return results_root / STABILITY_ANALYSIS_DIRNAME


def run_stability_analysis(
    results_root: str | Path,
    *,
    run_glob: str = "test_run_v*",
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    expected_runs: int = 10,
    skip_per_query: bool = False,
) -> dict[str, object]:
    results_root = resolve_results_root(Path(results_root))
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    run_dirs = discover_run_dirs(results_root, run_glob)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {results_root} matching {run_glob}")

    run_ids = [path.name for path in run_dirs]
    eval_paths = sorted(
        [path for run_dir in run_dirs for path in run_dir.rglob(f"*{EVAL_SUFFIX}")],
        key=lambda path: path.as_posix(),
    )
    log_paths = sorted(
        [path for run_dir in run_dirs for path in run_dir.rglob(f"*{LOG_SUFFIX}")],
        key=lambda path: path.as_posix(),
    )

    if not eval_paths:
        raise FileNotFoundError(f"No .eval files found under run directories in {results_root}")

    aggregate_groups, per_query_groups = collect_eval_values(eval_paths, results_root)
    log_groups = collect_log_values(log_paths, results_root)

    aggregate_key_fields = ["model", "method_family", "method", "topk", "condition", "metric"]
    per_query_key_fields = aggregate_key_fields + ["qid"]
    common_stat_fields = [
        "n_runs",
        "run_ids",
        "missing_run_ids",
        "mean",
        "sample_std",
        "sample_variance",
        "population_std",
        "population_variance",
        "std_error",
        "ci95_low",
        "ci95_high",
        "min",
        "max",
        "range",
        "coefficient_of_variation",
    ]

    aggregate_rows = group_to_rows(aggregate_groups, aggregate_key_fields, run_ids)
    log_rows = group_to_rows(log_groups, aggregate_key_fields, run_ids) if log_groups else []

    per_query_rows: list[dict[str, object]] = []
    per_query_summary_rows: list[dict[str, object]] = []
    if not skip_per_query:
        per_query_rows = group_to_rows(per_query_groups, per_query_key_fields, run_ids)
        per_query_summary_rows = build_per_query_summary(per_query_rows)

    report = build_report(
        results_root=results_root,
        run_ids=run_ids,
        aggregate_rows=aggregate_rows,
        per_query_summary_rows=per_query_summary_rows,
        log_rows=log_rows,
        primary_metric=primary_metric,
        expected_runs=expected_runs,
    )

    return {
        "results_root": results_root,
        "run_ids": run_ids,
        "eval_paths": eval_paths,
        "log_paths": log_paths,
        "aggregate_rows": aggregate_rows,
        "per_query_rows": per_query_rows,
        "per_query_summary_rows": per_query_summary_rows,
        "log_rows": log_rows,
        "report": report,
        "aggregate_key_fields": aggregate_key_fields,
        "per_query_key_fields": per_query_key_fields,
        "common_stat_fields": common_stat_fields,
    }


def main() -> None:
    args = parse_args()
    results_root = resolve_results_root(args.results_root)
    output_dir = args.output_dir or default_output_dir(results_root)

    try:
        analysis = run_stability_analysis(
            results_root,
            run_glob=args.run_glob,
            primary_metric=args.primary_metric,
            expected_runs=args.expected_runs,
            skip_per_query=args.skip_per_query,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    run_ids = analysis["run_ids"]
    aggregate_rows = analysis["aggregate_rows"]
    per_query_rows = analysis["per_query_rows"]
    per_query_summary_rows = analysis["per_query_summary_rows"]
    log_rows = analysis["log_rows"]
    aggregate_key_fields = analysis["aggregate_key_fields"]
    per_query_key_fields = analysis["per_query_key_fields"]
    common_stat_fields = analysis["common_stat_fields"]

    run_value_fields = list(run_ids)
    write_csv(
        output_dir / "evaluation_stability.csv",
        aggregate_rows,
        aggregate_key_fields + common_stat_fields + run_value_fields,
    )
    write_csv(
        output_dir / "log_stability.csv",
        log_rows,
        aggregate_key_fields + common_stat_fields + run_value_fields,
    )

    if not args.skip_per_query:
        write_csv(
            output_dir / "per_query_stability.csv",
            per_query_rows,
            per_query_key_fields + common_stat_fields + run_value_fields,
        )
        write_csv(
            output_dir / "per_query_stability_summary.csv",
            per_query_summary_rows,
            [
                "model",
                "method_family",
                "method",
                "topk",
                "condition",
                "metric",
                "n_queries",
                "mean_query_std",
                "median_query_std",
                "p90_query_std",
                "max_query_std",
                "mean_query_range",
                "median_query_range",
                "p90_query_range",
                "max_query_range",
                "queries_std_gt_0_001",
                "queries_std_gt_0_005",
                "queries_std_gt_0_01",
            ],
        )

    (output_dir / "STABILITY_REPORT.md").write_text(str(analysis["report"]))

    print(f"Runs discovered: {len(run_ids)}")
    print(f"Eval files parsed: {len(analysis['eval_paths'])}")
    print(f"Log files parsed: {len(analysis['log_paths'])}")
    print(f"Wrote stability analysis to {output_dir}")
    if len(run_ids) != args.expected_runs:
        print(f"WARNING: expected {args.expected_runs} runs, found {len(run_ids)}")


if __name__ == "__main__":
    main()
