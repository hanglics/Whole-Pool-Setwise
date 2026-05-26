#!/usr/bin/env bash
# submit_max_context_jobs.sh
#
# Submit the MaxContext stability-layout sbatch jobs with one command,
# parameterised by run tag, model, and dataset (DL19 / DL20). Everything else
# (BM25 run path, passage length, scoring mode, sort algorithms, output
# directory layout) stays fixed. Pool sizes default to {10,20,30,40,50}; use
# --pool-sizes to override.
#
# Usage:
#   ./submit_max_context_jobs.sh [--tag TAG] [--model MODEL]
#                                [--dataset DL19|DL20]
#                                [--idea007-only]
#                                [--maxcontext-only]
#                                [--shuffle|--reverse]
#                                [--overwrite]
#                                [--pool-sizes "10 20 30 40 50"]
#                                [--dry-run] [-h|--help]
#
# Examples:
#   ./submit_max_context_jobs.sh                                            # all defaults
#   ./submit_max_context_jobs.sh --tag test_run_v2 --model Qwen/Qwen3-8B
#   ./submit_max_context_jobs.sh --tag prod_v1 --model Qwen/Qwen3.5-9B \
#                                --dataset DL20
#   ./submit_max_context_jobs.sh --dry-run                                  # print only

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults (match the original Need_to_Run_Max_Context.txt baseline)
# -----------------------------------------------------------------------------
TAG="test_run_v1"
MODEL="Qwen/Qwen3-4B"
DATASET="DL19"
DRY_RUN=0
IDEA007_ONLY=0
MAXCONTEXT_ONLY=0
POOL_SIZES_OVERRIDE=""
SHUFFLE_FLAG=0
REVERSE_FLAG=0
OVERWRITE=0

usage() {
  cat <<'USAGE'
Usage: submit_max_context_jobs.sh [options]

Options:
  --tag TAG          Run-tag fragment to substitute for "test_run_v1" in
                     the output directory paths (default: test_run_v1).
  --model MODEL      HuggingFace model identifier, e.g. "Qwen/Qwen3-4B"
                     (default: Qwen/Qwen3-4B). The model tag in output
                     directories is derived as the lower-cased basename
                     after the final "/" (e.g. Qwen/Qwen3.5-9B -> qwen3.5-9b).
  --dataset NAME     One of DL19 or DL20 (default: DL19). Mapped to:
                       DL19 -> msmarco-passage/trec-dl-2019/judged
                       DL20 -> msmarco-passage/trec-dl-2020/judged
                     and the corresponding BM25 run path under runs/bm25/.
  --dry-run          Print sbatch commands instead of submitting them.
  --idea007-only     Submit only the historical 7-block IDEA_007 layout
                     (35 jobs with the default 5-pool sweep). Use this for
                     Phase C' byte-equality rechecks.
  --maxcontext-only  Submit only the three MaxContext blocks:
                     MaxContext-TopDown, MaxContext-DualEnd, and
                     MaxContext-BottomUp. Used by EMNLP Phase C through
                     submit_emnlp_stability_jobs.sh.
  --shuffle          MaxContext-only: per-round shuffle of the remaining pool
                     with fixed seed 929. Submits Blocks 5-7 only.
  --reverse          MaxContext-only: per-round reverse of the remaining pool.
                     Submits Blocks 5-7 only.
  --overwrite        Re-submit even if the target .txt result file already
                     exists and is non-empty. Default: skip with an [INFO]
                     message.
  --pool-sizes LIST  Override the default pool sizes with a whitespace-separated
                     positive-integer list, e.g. "10 20 30 40 50 100".
                     Omit this flag to preserve the canonical 5-pool layout.
  -h | --help        Show this help and exit.

By default, submits 55 sbatch jobs (11 method blocks x 5 pools):
  - Original TopDown-Heap   (WS=3)
  - Original TopDown-Bubble (WS=3)
  - Original TopDown-Heap   (WS=PS)
  - Original TopDown-Bubble (WS=PS)
  - MaxContext-TopDown
  - MaxContext-DualEnd     (writes to phase1/, all others write to baseline/)
  - MaxContext-BottomUp
  - Original BottomUp-Heap   (WS=3)
  - Original BottomUp-Bubble (WS=3)
  - Original BottomUp-Heap   (WS=PS)
  - Original BottomUp-Bubble (WS=PS)

With --idea007-only, suppresses the four BottomUp blocks and submits the
historical 35-job IDEA_007 layout.
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
    --dry-run)
      DRY_RUN=1; shift ;;
    --idea007-only)
      IDEA007_ONLY=1; shift ;;
    --maxcontext-only)
      MAXCONTEXT_ONLY=1; shift ;;
    --shuffle)
      SHUFFLE_FLAG=1; shift ;;
    --reverse)
      REVERSE_FLAG=1; shift ;;
    --overwrite)
      OVERWRITE=1; shift ;;
    --pool-sizes)
      [[ $# -ge 2 ]] || { echo "Error: --pool-sizes requires a value" >&2; exit 2; }
      POOL_SIZES_OVERRIDE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
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
# Resolve dataset shortcut -> dataset path, BM25 run path, dataset tag
# -----------------------------------------------------------------------------
case "$DATASET" in
  DL19|dl19)
    DATASET_PATH="msmarco-passage/trec-dl-2019/judged"
    BM25_RUN="runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"
    DATASET_TAG="dl19"
    ;;
  DL20|dl20)
    DATASET_PATH="msmarco-passage/trec-dl-2020/judged"
    BM25_RUN="runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt"
    DATASET_TAG="dl20"
    ;;
  *)
    echo "Error: --dataset must be DL19 or DL20 (got: '$DATASET')" >&2
    exit 2
    ;;
esac

# -----------------------------------------------------------------------------
# Derive model tag for the output-directory path: basename of the HF id, lowercased.
# Matches the convention used in Need_to_Run_Max_Context.txt:
#   Qwen/Qwen3-4B    -> qwen3-4b
#   Qwen/Qwen3.5-9B  -> qwen3.5-9b
# -----------------------------------------------------------------------------
MODEL_BASE="${MODEL##*/}"
MODEL_TAG="$(printf '%s' "$MODEL_BASE" | tr '[:upper:]' '[:lower:]')"

# Resolve conda env from model family. The submission helper passes this
# explicitly through sbatch --export because some clusters do not preserve the
# caller environment by default. ranker_env is the IDEA_007 default (Qwen3
# family + pyserini); qwen35_env covers Qwen3.5 / Llama-3.1 / Ministral-3.
case "$MODEL_BASE" in
  Qwen3.5-*|Meta-Llama-3.1-*|Ministral-3-*)
    export CONDA_ENV="${QWEN35_CONDA_ENV:-qwen35_env}" ;;
  *)
    export CONDA_ENV="${RANKER_CONDA_ENV:-ranker_env}" ;;
esac
export LLMRANKERS_OVERWRITE_RESULTS="$OVERWRITE"

OUT_PREFIX="results/maxcontext_dualend/${TAG}"
RUN_BASELINE="${OUT_PREFIX}/baseline/${MODEL_TAG}-${DATASET_TAG}"
RUN_PHASE1="${OUT_PREFIX}/phase1/${MODEL_TAG}-${DATASET_TAG}"

CONDITION_ACTIVE=0
CONDITION_SUFFIX=""
unset SHUFFLE REVERSE
if [[ "$SHUFFLE_FLAG" -eq 1 ]]; then
  CONDITION_ACTIVE=1
  CONDITION_SUFFIX="_shuffle"
  export SHUFFLE=1
elif [[ "$REVERSE_FLAG" -eq 1 ]]; then
  CONDITION_ACTIVE=1
  CONDITION_SUFFIX="_reverse"
  export REVERSE=1
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

# -----------------------------------------------------------------------------
# Submission helper: dry-runs print, real runs ensure the output directory
# exists then submit. Tracks total jobs submitted in $JOB_COUNT.
#
# Convention used by every block below: submit() receives the expected .txt
# filename first, then the launcher command. After that filename is shifted
# away, the launcher's 4th positional argument is OUTPUT_DIR, available here
# as $5. Pre-creating it here fails fast on permission / typo issues at
# submission time instead of hours later when the job runs.
# -----------------------------------------------------------------------------
JOB_COUNT=0
SKIP_COUNT=0

submit() {
  local expected_txt_name="${1:?submit() expects expected result filename as 1st arg}"
  shift
  local output_dir="${5:?submit() expects OUTPUT_DIR as the 4th launcher arg}"
  local expected_txt="${output_dir}/${expected_txt_name}"
  local export_vars="ALL,CONDA_ENV=${CONDA_ENV},ANALYSIS_LOG_DIR=${output_dir},LLMRANKERS_OVERWRITE_RESULTS=${OVERWRITE}"
  if [[ "$SHUFFLE_FLAG" -eq 1 ]]; then
    export_vars+=",SHUFFLE=1"
  elif [[ "$REVERSE_FLAG" -eq 1 ]]; then
    export_vars+=",REVERSE=1"
  fi
  if [[ -s "$expected_txt" && "$OVERWRITE" -eq 0 ]]; then
    echo "[INFO] Result file already exists: $expected_txt" >&2
    echo "[INFO] Skipping submission. If overwrite, add \"--overwrite\"." >&2
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return 0
  fi
  JOB_COUNT=$((JOB_COUNT + 1))
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'sbatch'
    printf ' --export=%s' "$export_vars"
    printf ' %s' "$@"
    printf '\n'
  else
    mkdir -p "$output_dir"
    sbatch --export="$export_vars" "$@"
  fi
}

cat <<INFO >&2
==> submit_max_context_jobs.sh
    tag         : ${TAG}
    model       : ${MODEL}     (model_tag = ${MODEL_TAG})
    dataset     : ${DATASET}   (path = ${DATASET_PATH}; tag = ${DATASET_TAG})
    bm25 run    : ${BM25_RUN}
    output base : ${OUT_PREFIX}
    maxctx only : $([[ $MAXCONTEXT_ONLY -eq 1 ]] && echo "yes" || echo "no")
    mode        : $([[ $DRY_RUN -eq 1 ]] && echo "DRY-RUN (no sbatch)" || echo "submitting")
INFO

if [[ "$CONDITION_ACTIVE" -eq 0 && "$MAXCONTEXT_ONLY" -eq 0 ]]; then
  # =============================================================================
  # Block 1 - Original TopDown-Heap (WS=3)   [num_child = 2]
  # =============================================================================
  for N in "${POOL_SIZES[@]}"; do
    submit "topdown_heapsort.txt" experiments/run_topdown_bigram.sh \
      "$MODEL" \
      "$DATASET_PATH" \
      "$BM25_RUN" \
      "${RUN_BASELINE}/original/ws-3/top${N}" \
      cuda generation 2 "$N" "$N" 512 heapsort
  done

  # =============================================================================
  # Block 2 - Original TopDown-Bubble (WS=3)  [num_child = 2]
  # =============================================================================
  for N in "${POOL_SIZES[@]}"; do
    submit "topdown_bubblesort.txt" experiments/run_topdown_bigram.sh \
      "$MODEL" \
      "$DATASET_PATH" \
      "$BM25_RUN" \
      "${RUN_BASELINE}/original/ws-3/top${N}" \
      cuda generation 2 "$N" "$N" 512 bubblesort
  done

  # =============================================================================
  # Block 3 - Original TopDown-Heap (WS=PS)  [num_child = pool_size - 1]
  # =============================================================================
  for N in "${POOL_SIZES[@]}"; do
    submit "topdown_heapsort.txt" experiments/run_topdown_bigram.sh \
      "$MODEL" \
      "$DATASET_PATH" \
      "$BM25_RUN" \
      "${RUN_BASELINE}/original/ws-ps/top${N}" \
      cuda generation $((N - 1)) "$N" "$N" 512 heapsort
  done

  # =============================================================================
  # Block 4 - Original TopDown-Bubble (WS=PS)  [num_child = pool_size - 1]
  # =============================================================================
  for N in "${POOL_SIZES[@]}"; do
    submit "topdown_bubblesort.txt" experiments/run_topdown_bigram.sh \
      "$MODEL" \
      "$DATASET_PATH" \
      "$BM25_RUN" \
      "${RUN_BASELINE}/original/ws-ps/top${N}" \
      cuda generation $((N - 1)) "$N" "$N" 512 bubblesort
  done
fi

# =============================================================================
# Block 5 - MaxContext-TopDown
# =============================================================================
for N in "${POOL_SIZES[@]}"; do
  submit "maxcontext_topdown.txt" experiments/run_maxcontext_topdown.sh \
    "$MODEL" \
    "$DATASET_PATH" \
    "$BM25_RUN" \
    "${RUN_BASELINE}/max-context/topdown/top${N}${CONDITION_SUFFIX}" \
    cuda generation "$N" 512
done

# =============================================================================
# Block 6 - MaxContext-DualEnd        (NOTE: writes under phase1/, not baseline/)
# =============================================================================
for N in "${POOL_SIZES[@]}"; do
  submit "maxcontext_dualend.txt" experiments/run_maxcontext_dualend.sh \
    "$MODEL" \
    "$DATASET_PATH" \
    "$BM25_RUN" \
    "${RUN_PHASE1}/top${N}${CONDITION_SUFFIX}" \
    cuda generation "$N" 512
done

# =============================================================================
# Block 7 - MaxContext-BottomUp
# =============================================================================
for N in "${POOL_SIZES[@]}"; do
  submit "maxcontext_bottomup.txt" experiments/run_maxcontext_bottomup.sh \
    "$MODEL" \
    "$DATASET_PATH" \
    "$BM25_RUN" \
    "${RUN_BASELINE}/max-context/bottomup/top${N}${CONDITION_SUFFIX}" \
    cuda generation "$N" 512
done


# ===========================================================================
# Block 8 - Original BottomUp-Heap (WS=3) [num_child = 2]
# ===========================================================================
if [[ "$IDEA007_ONLY" -eq 0 && "$CONDITION_ACTIVE" -eq 0 && "$MAXCONTEXT_ONLY" -eq 0 ]]; then
  for N in "${POOL_SIZES[@]}"; do
    submit "bottomup_heapsort.txt" experiments/run_bottomup_bigram.sh \
      "$MODEL" "$DATASET_PATH" "$BM25_RUN" \
      "${RUN_BASELINE}/original/bottomup/ws-3/top${N}" \
      cuda generation 2 "$N" "$N" 512 heapsort
  done

  # ===========================================================================
  # Block 9 - Original BottomUp-Bubble (WS=3) [num_child = 2]
  # ===========================================================================
  for N in "${POOL_SIZES[@]}"; do
    submit "bottomup_bubblesort.txt" experiments/run_bottomup_bigram.sh \
      "$MODEL" "$DATASET_PATH" "$BM25_RUN" \
      "${RUN_BASELINE}/original/bottomup/ws-3/top${N}" \
      cuda generation 2 "$N" "$N" 512 bubblesort
  done

  # ===========================================================================
  # Block 10 - Original BottomUp-Heap (WS=PS) [num_child = pool_size - 1]
  # ===========================================================================
  for N in "${POOL_SIZES[@]}"; do
    submit "bottomup_heapsort.txt" experiments/run_bottomup_bigram.sh \
      "$MODEL" "$DATASET_PATH" "$BM25_RUN" \
      "${RUN_BASELINE}/original/bottomup/ws-ps/top${N}" \
      cuda generation $((N - 1)) "$N" "$N" 512 heapsort
  done

  # ===========================================================================
  # Block 11 - Original BottomUp-Bubble (WS=PS) [num_child = pool_size - 1]
  # ===========================================================================
  for N in "${POOL_SIZES[@]}"; do
    submit "bottomup_bubblesort.txt" experiments/run_bottomup_bigram.sh \
      "$MODEL" "$DATASET_PATH" "$BM25_RUN" \
      "${RUN_BASELINE}/original/bottomup/ws-ps/top${N}" \
      cuda generation $((N - 1)) "$N" "$N" 512 bubblesort
  done
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "" >&2
  echo "==> DRY-RUN done. ${JOB_COUNT} sbatch jobs would have been submitted; ${SKIP_COUNT} skipped because result exists." >&2
else
  echo "" >&2
  echo "==> Submitted ${JOB_COUNT} sbatch jobs; skipped ${SKIP_COUNT} existing result(s)." >&2
fi
