#!/usr/bin/env bash
# Phase A EMNLP smoke gate: 3 models x 7 methods x dl19 x pools {50,100}.

set -euo pipefail

DRY_RUN=0
EVAL_ONLY=0
VERIFY_ONLY=0
OVERWRITE=0
TAG="phase_a"
OUTPUT_ROOT="results/emnlp/smoke/phase_a"

usage() {
  cat <<'USAGE'
Usage: scripts/smoke_emnlp_models.sh [--dry-run] [--eval-only] [--verify-only] [--overwrite]

Submits the 42 Phase A smoke jobs unless --eval-only or --verify-only is passed.
Outputs are written under results/emnlp/smoke/phase_a/{model_tag}/dl19/{method}/pool{50,100}/.

Use --eval-only after the jobs complete to write .eval files next to the .txt
outputs, then --verify-only to apply the method-aware smoke checks.

Runs on the login node in your interactive shell; activate ranker_env before
invoking eval-only or verify-only. Submit mode dispatches GPU SLURM jobs only.

  --overwrite   Submit-mode only: re-submit even if the target .txt result file
                already exists and is non-empty. Default: skip with an [INFO]
                message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

MODELS=(
  "Qwen/Qwen3.5-9B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "mistralai/Ministral-3-8B-Instruct-2512"
)

METHODS=(
  "topdown_bubblesort"
  "topdown_heapsort"
  "bottomup_bubblesort"
  "bottomup_heapsort"
  "maxcontext_topdown"
  "maxcontext_bottomup"
  "maxcontext_dualend"
)

SMOKE_POOLS=(50 100)
# dl19 has 43 judged queries; this script is dl19-only per Phase A.
N_QUERIES=43

model_tag() {
  printf '%s' "${1##*/}" | tr './' '-' | tr '[:upper:]' '[:lower:]'
}

verify_cell() {
  local model="$1"
  local method="$2"
  local pool_size="$3"
  local tag
  tag="$(model_tag "$model")"
  local dir="${OUTPUT_ROOT}/${tag}/dl19/${method}/pool${pool_size}"
  local txt="${dir}/${method}.txt"
  local eval_file="${dir}/${method}.eval"
  local log="${dir}/${method}.log"
  local expected_lines=$((N_QUERIES * pool_size))

  [[ -f "$txt" ]] || { echo "[FAIL] missing ${txt}" >&2; return 1; }
  [[ -f "$eval_file" ]] || { echo "[FAIL] missing ${eval_file}" >&2; return 1; }
  [[ -f "$log" ]] || { echo "[FAIL] missing ${log}" >&2; return 1; }

  local lines
  lines="$(wc -l < "$txt" | tr -d ' ')"
  [[ "$lines" == "$expected_lines" ]] || { echo "[FAIL] ${txt}: expected ${expected_lines} lines, got ${lines}" >&2; return 1; }

  python3 - "$txt" "$eval_file" "$N_QUERIES" <<'PY'
import sys
from collections import defaultdict
run_path, eval_path, expected_qids_raw = sys.argv[1:]
expected_qids = int(expected_qids_raw)
top_docs = defaultdict(list)
with open(run_path) as handle:
    for line in handle:
        parts = line.split()
        if len(parts) < 4:
            raise SystemExit(f"bad run line: {line!r}")
        qid, docid, rank = parts[0], parts[2], int(parts[3])
        if rank <= 10:
            top_docs[qid].append(docid)
if len(top_docs) != expected_qids:
    raise SystemExit(f"expected {expected_qids} qids, got {len(top_docs)}")
for qid, docs in top_docs.items():
    if len(docs) != 10 or len(set(docs)) != 10:
        raise SystemExit(f"qid {qid} top10 is not 10 distinct docids")
ndcg = None
with open(eval_path) as handle:
    for line in handle:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "ndcg_cut_10" and parts[1] == "all":
            ndcg = float(parts[2])
            break
if ndcg is None:
    raise SystemExit("missing ndcg_cut_10 all")
if ndcg <= 0:
    raise SystemExit(f"ndcg_cut_10 all is not positive: {ndcg}")
PY

  if grep -Eq "Traceback|ERROR|exceeds model limit" "$log"; then
    echo "[FAIL] ${log}: contains Traceback/ERROR/exceeds model limit" >&2
    return 1
  fi

	  case "$method" in
	    maxcontext_*)
	      if ! python3 - "$log" <<'PY'
import re
import sys

log_path = sys.argv[1]
text = open(log_path).read()


def avg_counter(label):
    match = re.search(rf"^Avg {re.escape(label)}: ([0-9]+(?:\.[0-9]+)?)$", text, flags=re.MULTILINE)
    if match is None:
        raise SystemExit(f"missing Avg {label} counter")
    return float(match.group(1))


def total_retries():
    match = re.search(r"^Total retries fired: ([0-9]+)$", text, flags=re.MULTILINE)
    return int(match.group(1)) if match is not None else 0


parse_fallbacks = avg_counter("parse fallbacks")
numeric_fallbacks = avg_counter("numeric out-of-range fallbacks")
degenerate_fallbacks = avg_counter("degenerate repetition fallbacks")
unparseable_fallbacks = avg_counter("unparseable after exhaustion fallbacks")
duplicate_fallbacks = avg_counter("duplicate label fallbacks")
strict_failures = avg_counter("parse_failure_strict")
bm25_fallbacks = avg_counter("parse_failure_bm25_fallback")

if strict_failures != 0:
    raise SystemExit("parse_failure_strict counter is non-zero")
if bm25_fallbacks != 0:
    raise SystemExit("parse_failure_bm25_fallback counter is non-zero")

accepted_fallbacks = (
    parse_fallbacks
    + numeric_fallbacks
    + degenerate_fallbacks
    + unparseable_fallbacks
    + duplicate_fallbacks
)
if accepted_fallbacks > 0 and total_retries() < 3:
    raise SystemExit(
        "accepted MaxContext fallbacks require Total retries fired >= 3 "
        f"(got {total_retries()})"
    )
PY
	      then
	        echo "[FAIL] ${log}: MaxContext fallback/retry gate failed" >&2
	        return 1
	      fi
	      # Smoke gate: parse_failure_strict must be exactly 0. Any non-zero value
	      # is an unrecovered parse failure under strict mode (set
	      # _allow_parse_failure_bm25_fallback=False for smoke). Phase B/C/F dispatchers
	      # may flip the fallback on, in which case parse_failure_bm25_fallback is
	      # non-zero and parse_failure_strict stays at 0 — but for Phase A smoke we
	      # require both to be zero so we surface new failure modes.
	      ;;
	  esac

  echo "[OK] ${tag} ${method} pool${pool_size}"
}

print_summary() {
  echo "[smoke] processed ${#MODELS[@]} × ${#METHODS[@]} × ${#SMOKE_POOLS[@]} = $((${#MODELS[@]} * ${#METHODS[@]} * ${#SMOKE_POOLS[@]})) cells"
}

if [[ "$VERIFY_ONLY" -eq 0 && "$EVAL_ONLY" -eq 0 ]]; then
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      for ps in "${SMOKE_POOLS[@]}"; do
        args=(
          bash submit_emnlp_jobs.sh
          --method "$method"
          --model "$model"
          --dataset dl19
          --pool-size "$ps"
          --tag "$TAG"
          --output-root "$OUTPUT_ROOT"
        )
        if [[ "$DRY_RUN" -eq 1 ]]; then
          args+=(--dry-run)
        fi
        if [[ "$OVERWRITE" -eq 1 ]]; then
          args+=(--overwrite)
        fi
        "${args[@]}"
      done
    done
  done
  print_summary
fi

if [[ "$EVAL_ONLY" -eq 1 ]]; then
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      for ps in "${SMOKE_POOLS[@]}"; do
        args=(
          bash eval_emnlp_jobs.sh
          --method "$method"
          --model "$model"
          --dataset dl19
          --pool-size "$ps"
          --tag "$TAG"
          --output-root "$OUTPUT_ROOT"
        )
        if [[ "$DRY_RUN" -eq 1 ]]; then
          args+=(--dry-run)
        fi
        "${args[@]}"
      done
    done
  done
  print_summary
fi

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  failures=0
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      for ps in "${SMOKE_POOLS[@]}"; do
        verify_cell "$model" "$method" "$ps" || failures=$((failures + 1))
      done
    done
  done
  [[ "$failures" -eq 0 ]] || exit 1
  print_summary
fi
