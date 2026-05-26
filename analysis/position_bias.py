"""
Position Bias Analysis (§5.4)

Analyzes whether LLMs exhibit position bias when selecting best/worst passages.
Uses comparison logs produced by running with --log_comparisons flag.

Usage:
    # Step 1: Generate comparison logs (re-run with logging)
    python run.py run --model_name_or_path google/flan-t5-xl \
        --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
        --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
        --save_path results/analysis/topdown_heapsort.txt \
        --scoring generation --hits 100 --passage_length 128 \
        --log_comparisons results/analysis/topdown_heapsort_comparisons.jsonl \
        setwise --num_child 3 --method heapsort --k 10 --direction topdown

    # Step 2: Analyze
    python analysis/position_bias.py \
        --log results/analysis/topdown_heapsort_comparisons.jsonl
"""

import argparse
import json
from collections import defaultdict, Counter


def load_comparison_entries(log_paths):
    all_entries = []
    for log_path in log_paths:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))
    return all_entries


def summarize_position_bias(all_entries):
    if not all_entries:
        return None

    schemes = {entry.get("label_scheme", "letters_a_w") for entry in all_entries}
    if len(schemes) > 1:
        raise ValueError(
            f"Mixed label_scheme values in input logs: {schemes}. "
            "Run analysis separately per scheme."
        )
    scheme = schemes.pop()

    by_type = defaultdict(list)
    for entry in all_entries:
        by_type[entry["type"]].append(entry)

    def render_label(entries, index: int) -> str:
        for entry in entries:
            if index < len(entry["positions"]):
                return entry["positions"][index]
        return ""

    type_summaries = {}
    for comp_type, entries in sorted(by_type.items()):
        n_positions = max(len(e["positions"]) for e in entries)
        position_counts = Counter()
        total = 0

        for entry in entries:
            selected = entry["selected"]
            if selected and selected in entry["positions"]:
                idx = entry["positions"].index(selected)
                position_counts[idx] += 1
                total += 1

        selection_rows = []
        chi2 = None
        expected_count = None
        if total > 0:
            expected_count = total / n_positions
            chi2 = sum(
                (position_counts.get(i, 0) - expected_count) ** 2 / expected_count
                for i in range(n_positions)
            )
            for i in range(n_positions):
                count = position_counts.get(i, 0)
                freq = count / total
                expected = 1.0 / n_positions
                bias = "▲" if freq > expected * 1.3 else ("▼" if freq < expected * 0.7 else " ")
                selection_rows.append(
                    {
                        "comparison_type": comp_type,
                        "position_index": i,
                        "label": render_label(entries, i),
                        "count": count,
                        "frequency": freq,
                        "expected": expected,
                        "bias": bias,
                    }
                )

        accuracy = None
        accuracy_rows = []
        if any("doc_relevances" in e for e in entries):
            correct = 0
            total_with_rels = 0
            pos_totals = Counter()
            pos_correct = Counter()
            for entry in entries:
                rels = entry.get("doc_relevances")
                if rels is None:
                    continue
                selected = entry["selected"]
                if selected and selected in entry["positions"]:
                    idx = entry["positions"].index(selected)
                    is_correct = False
                    if comp_type in ("best", "dual_best"):
                        is_correct = rels[idx] == max(rels)
                    elif comp_type in ("worst", "dual_worst"):
                        is_correct = rels[idx] == min(rels)
                    if is_correct:
                        correct += 1
                        pos_correct[idx] += 1
                    pos_totals[idx] += 1
                    total_with_rels += 1

            if total_with_rels > 0:
                accuracy = {
                    "correct": correct,
                    "total": total_with_rels,
                    "accuracy": correct / total_with_rels,
                }
                for i in range(n_positions):
                    if pos_totals[i]:
                        accuracy_rows.append(
                            {
                                "comparison_type": comp_type,
                                "position_index": i,
                                "label": render_label(entries, i),
                                "correct": pos_correct[i],
                                "total": pos_totals[i],
                                "accuracy": pos_correct[i] / pos_totals[i],
                            }
                        )

        type_summaries[comp_type] = {
            "entries": entries,
            "n_comparisons": len(entries),
            "n_positions": n_positions,
            "valid_selections": total,
            "selection_rows": selection_rows,
            "chi2": chi2,
            "expected_count": expected_count,
            "accuracy": accuracy,
            "accuracy_rows": accuracy_rows,
        }

    return {
        "total_comparisons": len(all_entries),
        "label_scheme": scheme,
        "type_counts": {t: len(v) for t, v in sorted(by_type.items())},
        "types": type_summaries,
    }


def render_position_bias_summary(summary):
    lines = []

    def out(s=""):
        lines.append(s)

    out("=" * 70)
    out("Position Bias Analysis")
    out(f"Total comparisons: {summary['total_comparisons']}")
    out(f"Label scheme: {summary['label_scheme']}")
    out(f"Types: {', '.join(f'{t}={v}' for t, v in summary['type_counts'].items())}")
    out("=" * 70)

    for comp_type, data in summary["types"].items():
        out(f"\n--- Type: {comp_type} ({data['n_comparisons']} comparisons) ---")

        if data["valid_selections"] == 0:
            out("  No valid selections found.")
            continue

        out(f"\n  Selection frequency by position (total={data['valid_selections']}):")
        out(f"  {'Position':<12} {'Label':<8} {'Count':>8} {'Freq':>8} {'Expected':>10}")
        out(f"  {'-'*50}")

        for row in data["selection_rows"]:
            out(
                f"  {row['position_index']:<12} {row['label']:<8} "
                f"{row['count']:>8} {row['frequency']:>8.3f} "
                f"{row['expected']:>10.3f} {row['bias']}"
            )

        out(f"\n  Chi-squared: {data['chi2']:.2f} (df={data['n_positions']-1})")
        if data["expected_count"] < 5:
            out("  Note: expected counts are below 5, so the chi-squared approximation is unreliable.")

        if data["accuracy"]:
            acc = data["accuracy"]
            out(f"\n  Accuracy (selected optimal): {acc['correct']}/{acc['total']} = {acc['accuracy']:.3f}")
            out("  Accuracy by position:")
            for row in data["accuracy_rows"]:
                out(
                    f"    Position {row['position_index']} ({row['label']}): "
                    f"{row['correct']}/{row['total']} = {row['accuracy']:.3f}"
                )

    return lines


def main():
    parser = argparse.ArgumentParser(description="Position bias analysis from comparison logs")
    parser.add_argument("--log", required=True, nargs="+",
                        help="One or more comparison log files (.jsonl)")
    parser.add_argument("--output", default=None,
                        help="Save results to file (optional)")
    args = parser.parse_args()

    all_entries = load_comparison_entries(args.log)
    if not all_entries:
        print("No comparison entries found.")
        return

    summary = summarize_position_bias(all_entries)
    lines = render_position_bias_summary(summary)
    for line in lines:
        print(line)

    if args.output:
        with open(args.output, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
