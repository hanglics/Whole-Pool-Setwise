#!/usr/bin/env bash
# Submit EMNLP Phase C repeated-run stability jobs for MaxContext methods only.

set -euo pipefail

MODEL=""
DATASET="DL19"
TAG_PREFIX="emnlp_phase_c_required"
REPS=5
DRY_RUN=0
POOL_SIZES="100"
SHUFFLE=0
REVERSE=0
OVERWRITE=0

usage() {
  cat <<'USAGE'
Usage: submit_emnlp_stability_jobs.sh --model HF_ID [options]

Options:
  --model HF_ID       HuggingFace model id to run.
  --dataset DL19|DL20 Dataset shortcut passed through to submit_max_context_jobs.sh
                      (default: DL19; Phase C uses DL19).
  --tag-prefix TAG    Prefix under results/maxcontext_dualend/
                      (default: emnlp_phase_c_required).
  --reps N            Number of repeated runs (default: 5).
  --pool-sizes LIST   Whitespace-separated pool-size list passed through to
                      submit_max_context_jobs.sh (default: 100).
  --shuffle           MaxContext-only: pass through per-round shuffle.
  --reverse           MaxContext-only: pass through per-round reverse.
  --dry-run           Print sbatch commands only.
  --overwrite         Re-submit even if the target .txt result file already
                      exists and is non-empty. Default: skip with an [INFO]
                      message.
  -h | --help         Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) [[ $# -ge 2 ]] || { echo "Error: --model requires a value" >&2; exit 2; }; MODEL="$2"; shift 2 ;;
    --dataset) [[ $# -ge 2 ]] || { echo "Error: --dataset requires a value" >&2; exit 2; }; DATASET="$2"; shift 2 ;;
    --tag-prefix) [[ $# -ge 2 ]] || { echo "Error: --tag-prefix requires a value" >&2; exit 2; }; TAG_PREFIX="$2"; shift 2 ;;
    --reps) [[ $# -ge 2 ]] || { echo "Error: --reps requires a value" >&2; exit 2; }; REPS="$2"; shift 2 ;;
    --pool-sizes) [[ $# -ge 2 ]] || { echo "Error: --pool-sizes requires a value" >&2; exit 2; }; POOL_SIZES="$2"; shift 2 ;;
    --shuffle) SHUFFLE=1; shift ;;
    --reverse) REVERSE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "Error: --model is required" >&2; usage >&2; exit 2; }
if [[ "$SHUFFLE" -eq 1 && "$REVERSE" -eq 1 ]]; then
  echo "Error: --shuffle and --reverse are mutually exclusive" >&2
  exit 2
fi
[[ "$REPS" =~ ^[0-9]+$ && "$REPS" -gt 0 ]] || { echo "Error: --reps must be a positive integer" >&2; exit 2; }
read -ra POOL_SIZE_CHECK <<< "$POOL_SIZES"
[[ ${#POOL_SIZE_CHECK[@]} -gt 0 ]] || { echo "Error: --pool-sizes must not be empty" >&2; exit 2; }
for ps in "${POOL_SIZE_CHECK[@]}"; do
  [[ "$ps" =~ ^[1-9][0-9]*$ ]] || { echo "Error: --pool-sizes entry '$ps' is not a positive integer" >&2; exit 2; }
done

case "$DATASET" in
  DL19|dl19) DATASET_TAG="dl19" ;;
  DL20|dl20) DATASET_TAG="dl20" ;;
  *) echo "Error: --dataset must be DL19 or DL20" >&2; exit 2 ;;
esac

# Keep dots in the stability model tag to match submit_max_context_jobs.sh.
STABILITY_MODEL_TAG="$(printf '%s' "${MODEL##*/}" | tr '[:upper:]' '[:lower:]')"
for ((v = 1; v <= REPS; v++)); do
  CMD=(
    bash submit_max_context_jobs.sh
    --maxcontext-only
    --pool-sizes "$POOL_SIZES"
    --tag "${TAG_PREFIX}/${STABILITY_MODEL_TAG}-${DATASET_TAG}/stability-test-runs/test_run_v${v}"
    --model "$MODEL"
    --dataset "$DATASET"
  )
  if [[ "$SHUFFLE" -eq 1 ]]; then
    CMD+=(--shuffle)
  elif [[ "$REVERSE" -eq 1 ]]; then
    CMD+=(--reverse)
  fi
  if [[ "$OVERWRITE" -eq 1 ]]; then
    CMD+=(--overwrite)
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    CMD+=(--dry-run)
  fi
  "${CMD[@]}"
done
