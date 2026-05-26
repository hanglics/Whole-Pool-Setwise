#!/usr/bin/env bash
# eval_max_context_jobs.sh
#
# Evaluate every result file produced by submit_max_context_jobs.sh under a
# given (tag, model, dataset) using `pyserini.eval.trec_eval` with the same
# metric set as the user's example:
#
#   python -m pyserini.eval.trec_eval -q -l 2 \
#       -m ndcg_cut.3,5,10,20,30,40,50 dl19-passage <result>.txt
#
# For each result `<dir>/<name>.txt`, the per-query and aggregate metric rows
# from the trec_eval output are saved to `<dir>/<name>.eval`.
#
# Usage:
#   ./eval_max_context_jobs.sh [--tag TAG] [--model MODEL]
#                              [--dataset DL19|DL20]
#                              [--idea007-only]
#                              [--maxcontext-only]
#                              [--shuffle|--reverse]
#                              [--pool-sizes "10 20 30 40 50"]
#                              [--force] [--dry-run] [-h|--help]
#
# Examples:
#   ./eval_max_context_jobs.sh                                    # all defaults
#   ./eval_max_context_jobs.sh --tag test_run_v2 --model Qwen/Qwen3-8B
#   ./eval_max_context_jobs.sh --force                            # rebuild every .eval
#   ./eval_max_context_jobs.sh --dry-run                          # show targets only

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults (match submit_max_context_jobs.sh)
# -----------------------------------------------------------------------------
TAG="test_run_v1"
MODEL="Qwen/Qwen3-4B"
DATASET="DL19"
DRY_RUN=0
FORCE=0
IDEA007_ONLY=0
MAXCONTEXT_ONLY=0
POOL_SIZES_OVERRIDE=""
SHUFFLE_FLAG=0
REVERSE_FLAG=0

usage() {
  cat <<'USAGE'
Usage: eval_max_context_jobs.sh [options]

Options:
  --tag TAG          Run-tag fragment (default: test_run_v1). Must match the
                     value passed to submit_max_context_jobs.sh.
  --model MODEL      HuggingFace model identifier (default: Qwen/Qwen3-4B).
                     Used to locate the model-dataset subdirectory; the model
                     tag is the lower-cased basename after the final "/".
  --dataset NAME     One of DL19 or DL20 (default: DL19). Mapped to qrels:
                       DL19 -> dl19-passage
                       DL20 -> dl20-passage
  --force            Re-evaluate even if a .eval file already exists.
  --idea007-only     Evaluate only the historical 7-block IDEA_007 layout
                     (35 targets with the default 5-pool sweep). Use this for
                     Phase C' byte-equality rechecks.
  --maxcontext-only  Evaluate only the three MaxContext blocks:
                     MaxContext-TopDown, MaxContext-DualEnd, and
                     MaxContext-BottomUp. Used by EMNLP Phase C through
                     submit_emnlp_stability_jobs.sh.
  --shuffle          Evaluate MaxContext-only per-round shuffle outputs.
  --reverse          Evaluate MaxContext-only per-round reverse outputs.
  --pool-sizes LIST  Override the default pool sizes with a whitespace-separated
                     positive-integer list, e.g. "10 20 30 40 50 100".
                     Omit this flag to preserve the canonical 5-pool layout.
  --dry-run          Print the planned eval commands and skip targets without
                     actually invoking pyserini.eval.trec_eval.
  -h | --help        Show this help and exit.

For each expected result file produced by submit_max_context_jobs.sh under the
given (tag, model, dataset), this script runs:

    python -m pyserini.eval.trec_eval -q -l 2 \
        -m ndcg_cut.3,5,10,20,30,40,50 <qrels> <result>.txt

and saves the per-query + aggregate metric rows to <result>.eval next to the
source .txt file. Missing result files are logged but do not abort the run.

Runs on the login node in your interactive shell; activate ranker_env before invoking.
USAGE
}

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      [[ $# -ge 2 ]] || { echo "Error: --tag requires a value" >&2; exit 2; }
      TAG="$2"; shift 2 ;;
    --model)
      [[ $# -ge 2 ]] || { echo "Error: --model requires a value" >&2; exit 2; }
      MODEL="$2"; shift 2 ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "Error: --dataset requires a value" >&2; exit 2; }
      DATASET="$2"; shift 2 ;;
    --idea007-only) IDEA007_ONLY=1; shift ;;
    --maxcontext-only) MAXCONTEXT_ONLY=1; shift ;;
    --shuffle) SHUFFLE_FLAG=1; shift ;;
    --reverse) REVERSE_FLAG=1; shift ;;
    --pool-sizes)
      [[ $# -ge 2 ]] || { echo "Error: --pool-sizes requires a value" >&2; exit 2; }
      POOL_SIZES_OVERRIDE="$2"; shift 2 ;;
    --force)   FORCE=1;   shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "$SHUFFLE_FLAG" -eq 1 && "$REVERSE_FLAG" -eq 1 ]]; then
  echo "Error: --shuffle and --reverse are mutually exclusive" >&2
  exit 2
fi
if [[ "$IDEA007_ONLY" -eq 1 && "$MAXCONTEXT_ONLY" -eq 1 ]]; then
  echo "Error: --idea007-only and --maxcontext-only are mutually exclusive" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Resolve dataset shortcut -> qrels label + dataset tag
# -----------------------------------------------------------------------------
case "$DATASET" in
  DL19|dl19) QRELS="dl19-passage"; DATASET_TAG="dl19" ;;
  DL20|dl20) QRELS="dl20-passage"; DATASET_TAG="dl20" ;;
  *)
    echo "Error: --dataset must be DL19 or DL20 (got: '$DATASET')" >&2
    exit 2 ;;
esac

MODEL_BASE="${MODEL##*/}"
MODEL_TAG="$(printf '%s' "$MODEL_BASE" | tr '[:upper:]' '[:lower:]')"

OUT_PREFIX="results/maxcontext_dualend/${TAG}"
RUN_BASELINE="${OUT_PREFIX}/baseline/${MODEL_TAG}-${DATASET_TAG}"
RUN_PHASE1="${OUT_PREFIX}/phase1/${MODEL_TAG}-${DATASET_TAG}"

CONDITION_ACTIVE=0
CONDITION_SUFFIX=""
if [[ "$SHUFFLE_FLAG" -eq 1 ]]; then
  CONDITION_ACTIVE=1
  CONDITION_SUFFIX="_shuffle"
elif [[ "$REVERSE_FLAG" -eq 1 ]]; then
  CONDITION_ACTIVE=1
  CONDITION_SUFFIX="_reverse"
fi

POOL_SIZES=(10 20 30 40 50)
if [[ -n "${POOL_SIZES_OVERRIDE}" ]]; then
  read -ra POOL_SIZES <<< "$POOL_SIZES_OVERRIDE"
  [[ ${#POOL_SIZES[@]} -gt 0 ]] || {
    echo "Error: --pool-sizes resolved to an empty array; supply at least one positive integer" >&2
    exit 2
  }
  for ps in "${POOL_SIZES[@]}"; do
    [[ "$ps" =~ ^[1-9][0-9]*$ ]] || {
      echo "Error: --pool-sizes entry '$ps' is not a positive integer" >&2
      exit 2
    }
  done
fi

extract_eval_body() {
  local output="$1"
  local body
  body="$(printf '%s\n' "$output" | awk '/^Results:[[:space:]]*$/{found=1; next} found')"
  if [[ -n "$body" ]]; then
    printf '%s\n' "$body"
    return 0
  fi

  # Newer pyserini versions print trec_eval rows directly without a
  # "Results:" marker. Keep only metric rows and drop any warnings/log lines.
  printf '%s\n' "$output" | awk '
    NF >= 3 &&
    $1 ~ /^[[:alpha:]_][[:alnum:]_.-]*$/ &&
    $NF ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ { print }
  '
}

cat <<INFO >&2
==> eval_max_context_jobs.sh
    tag         : ${TAG}
    model       : ${MODEL}     (model_tag = ${MODEL_TAG})
    dataset     : ${DATASET}   (qrels = ${QRELS}; tag = ${DATASET_TAG})
    output base : ${OUT_PREFIX}
    maxctx only : $([[ $MAXCONTEXT_ONLY -eq 1 ]] && echo "yes" || echo "no")
    mode        : $([[ $DRY_RUN -eq 1 ]] && echo "DRY-RUN (no trec_eval calls)" || echo "evaluating")
    force       : $([[ $FORCE -eq 1 ]] && echo "yes (overwrite existing .eval)" || echo "no  (skip existing)")
INFO

# -----------------------------------------------------------------------------
# Build the list of expected result files. Mirrors the blocks of
# submit_max_context_jobs.sh in order.
# -----------------------------------------------------------------------------
EXPECTED=()

if [[ "$CONDITION_ACTIVE" -eq 0 && "$MAXCONTEXT_ONLY" -eq 0 ]]; then
  # Block 1 - Original TopDown-Heap   (WS=3)
  for N in "${POOL_SIZES[@]}"; do
    EXPECTED+=("${RUN_BASELINE}/original/ws-3/top${N}/topdown_heapsort.txt")
  done
  # Block 2 - Original TopDown-Bubble (WS=3)
  for N in "${POOL_SIZES[@]}"; do
    EXPECTED+=("${RUN_BASELINE}/original/ws-3/top${N}/topdown_bubblesort.txt")
  done
  # Block 3 - Original TopDown-Heap   (WS=PS)
  for N in "${POOL_SIZES[@]}"; do
    EXPECTED+=("${RUN_BASELINE}/original/ws-ps/top${N}/topdown_heapsort.txt")
  done
  # Block 4 - Original TopDown-Bubble (WS=PS)
  for N in "${POOL_SIZES[@]}"; do
    EXPECTED+=("${RUN_BASELINE}/original/ws-ps/top${N}/topdown_bubblesort.txt")
  done
fi
# Block 5 - MaxContext-TopDown
for N in "${POOL_SIZES[@]}"; do
  EXPECTED+=("${RUN_BASELINE}/max-context/topdown/top${N}${CONDITION_SUFFIX}/maxcontext_topdown.txt")
done
# Block 6 - MaxContext-DualEnd      (under phase1/, not baseline/)
for N in "${POOL_SIZES[@]}"; do
  EXPECTED+=("${RUN_PHASE1}/top${N}${CONDITION_SUFFIX}/maxcontext_dualend.txt")
done
# Block 7 - MaxContext-BottomUp
for N in "${POOL_SIZES[@]}"; do
  EXPECTED+=("${RUN_BASELINE}/max-context/bottomup/top${N}${CONDITION_SUFFIX}/maxcontext_bottomup.txt")
done

if [[ "$IDEA007_ONLY" -eq 0 && "$CONDITION_ACTIVE" -eq 0 && "$MAXCONTEXT_ONLY" -eq 0 ]]; then
  for N in "${POOL_SIZES[@]}"; do
    EXPECTED+=("${RUN_BASELINE}/original/bottomup/ws-3/top${N}/bottomup_heapsort.txt")
    EXPECTED+=("${RUN_BASELINE}/original/bottomup/ws-3/top${N}/bottomup_bubblesort.txt")
    EXPECTED+=("${RUN_BASELINE}/original/bottomup/ws-ps/top${N}/bottomup_heapsort.txt")
    EXPECTED+=("${RUN_BASELINE}/original/bottomup/ws-ps/top${N}/bottomup_bubblesort.txt")
  done
fi

# -----------------------------------------------------------------------------
# Counters
# -----------------------------------------------------------------------------
TOTAL=${#EXPECTED[@]}
EVAL_OK=0
EVAL_FAIL=0
SKIPPED_EXISTS=0
SKIPPED_MISSING=0
PLANNED=0

for RESULT in "${EXPECTED[@]}"; do
  EVAL_FILE="${RESULT%.txt}.eval"

  if [[ ! -f "$RESULT" ]]; then
    echo "[MISSING] $RESULT (job not completed?)" >&2
    SKIPPED_MISSING=$((SKIPPED_MISSING + 1))
    continue
  fi

  if [[ -f "$EVAL_FILE" && "$FORCE" -ne 1 ]]; then
    echo "[skip]    $EVAL_FILE already exists (--force to overwrite)" >&2
    SKIPPED_EXISTS=$((SKIPPED_EXISTS + 1))
    continue
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "python -m pyserini.eval.trec_eval -q -l 2 -m ndcg_cut.3,5,10,20,30,40,50 ${QRELS} ${RESULT}"
    echo "  --> ${EVAL_FILE}"
    PLANNED=$((PLANNED + 1))
    continue
  fi

  echo "[eval]    ${RESULT}" >&2

  # Capture stdout+stderr together because pyserini versions differ in where
  # they write trec_eval output.
  if OUTPUT="$(python -m pyserini.eval.trec_eval -q -l 2 \
                  -m ndcg_cut.3,5,10,20,30,40,50 \
                  "${QRELS}" "${RESULT}" 2>&1)"; then
    BODY="$(extract_eval_body "$OUTPUT")"
    if [[ -z "$BODY" ]]; then
      echo "[fail]    no trec_eval metric rows for $RESULT" >&2
      printf '%s\n' "$OUTPUT" >&2
      EVAL_FAIL=$((EVAL_FAIL + 1))
      continue
    fi
    printf '%s\n' "$BODY" > "$EVAL_FILE"
    EVAL_OK=$((EVAL_OK + 1))
  else
    EXIT_CODE=$?
    echo "[fail]    trec_eval exit ${EXIT_CODE} for $RESULT" >&2
    printf '%s\n' "$OUTPUT" >&2
    EVAL_FAIL=$((EVAL_FAIL + 1))
  fi
done

echo "" >&2
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "==> DRY-RUN summary: would evaluate ${PLANNED}/${TOTAL} files" >&2
  echo "    (missing on disk: ${SKIPPED_MISSING}; already-evaluated: ${SKIPPED_EXISTS})" >&2
else
  echo "==> Eval summary (out of ${TOTAL} expected result files):" >&2
  echo "    evaluated OK   : ${EVAL_OK}" >&2
  echo "    failed         : ${EVAL_FAIL}" >&2
  echo "    skipped (exists): ${SKIPPED_EXISTS}" >&2
  echo "    missing on disk: ${SKIPPED_MISSING}" >&2
  if [[ "$EVAL_FAIL" -gt 0 ]]; then
    exit 1
  fi
fi
