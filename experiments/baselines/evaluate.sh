#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN=$PYTHON
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=$(command -v python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=$(command -v python)
else
  echo "No Python interpreter found. Export PYTHON=/absolute/path/to/python." >&2
  exit 127
fi
[[ -x "$PYTHON_BIN" ]] || {
  echo "Selected Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 127
}

ROOT="results/paper"
OVERWRITE=0
INCLUDE_SMOKE=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT=$2; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --include-smoke) INCLUDE_SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: $0 [--root DIR] [--overwrite] [--include-smoke] [--dry-run]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

while IFS= read -r latest_file; do
  if [[ "$INCLUDE_SMOKE" != 1 && "$latest_file" == */smoke-q*/* ]]; then
    continue
  fi
  condition=$(dirname "$latest_file")
  attempt_name=$(<"$latest_file")
  [[ -n "$attempt_name" && "$attempt_name" != */* ]] || {
    echo "Invalid LATEST pointer: $latest_file" >&2
    exit 2
  }
  attempt="$condition/$attempt_name"
  [[ -f "$attempt/DONE" ]] || {
    echo "LATEST does not point to a complete attempt: $latest_file" >&2
    exit 2
  }
  manifest="$attempt/protocol_manifest.json"
  [[ -s "$manifest" ]] || { echo "Missing manifest: $attempt" >&2; exit 2; }
  metadata=$("$PYTHON_BIN" - "$manifest" "$attempt" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
attempt = Path(sys.argv[2])
record = json.loads(manifest_path.read_text(encoding="utf-8"))
if "output_objective" not in record or "output_depth" not in record:
    raise SystemExit(f"manifest lacks explicit objective/depth: {manifest_path}")

run_name = record.get("run_file")
if run_name is None:
    candidates = sorted(
        path for path in attempt.glob("*.txt") if path.name != "debug.txt"
    )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise SystemExit(
            f"expected exactly one non-sidecar TREC run in {attempt}; found: {names}"
        )
    run_name = candidates[0].name
if (
    not isinstance(run_name, str)
    or Path(run_name).name != run_name
    or Path(run_name).suffix != ".txt"
    or run_name == "debug.txt"
):
    raise SystemExit(f"invalid manifest run_file in {manifest_path}: {run_name!r}")
run_path = attempt / run_name
if not run_path.is_file() or run_path.stat().st_size == 0:
    raise SystemExit(f"missing or empty manifest TREC run: {run_path}")

print(
    record["dataset"],
    record["output_objective"],
    record["output_depth"],
    run_name,
    sep="\t",
)
PY
)
  IFS=$'\t' read -r dataset objective depth run_name <<< "$metadata"
  [[ "$depth" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid output_depth in $manifest: $depth" >&2
    exit 2
  }
  case "$dataset" in
    dl19) qrels=dl19-passage ;;
    dl20) qrels=dl20-passage ;;
    *) echo "Unsupported experiment dataset in $manifest: $dataset" >&2; exit 2 ;;
  esac
  run_file="$attempt/$run_name"
  stale_debug_eval="$attempt/debug.eval"
  if [[ -e "$stale_debug_eval" ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then
      echo "[DRY-RUN] rm -f $stale_debug_eval"
    else
      rm -f "$stale_debug_eval"
      echo "[remove] $stale_debug_eval"
    fi
  fi
  eval_file="${run_file%.txt}.eval"
  if [[ -s "$eval_file" && "$OVERWRITE" != 1 ]]; then
    echo "[skip] $eval_file"
    continue
  fi
  metrics=(-m ndcg_cut.10)
  if [[ "$objective" == full ]]; then
    metrics=(-m "ndcg_cut.10,$depth" -m "map_cut.$depth")
  fi
  cmd=("$PYTHON_BIN" -m pyserini.eval.trec_eval -q -l 2 "${metrics[@]}" "$qrels" "$run_file")
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[DRY-RUN]'; printf ' %q' "${cmd[@]}"; printf ' | tee %q\n' "$eval_file"
  else
    "${cmd[@]}" | tee "$eval_file"
  fi
done < <(find "$ROOT" -type f -name LATEST | sort)
