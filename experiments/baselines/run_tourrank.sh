#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "Usage: $0 MODEL DATASET LOCAL_RUN ATTEMPT_DIR TOURNAMENTS" >&2
  exit 2
fi
MODEL=$1
DATASET=$2
LOCAL_RUN=$3
ATTEMPT_DIR=$4
TOURNAMENTS=$5
SOURCE_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SOURCE_SCRIPT_DIR/../.." && pwd)}
BASELINES_DIR="$PROJECT_ROOT/experiments/baselines"

source "$BASELINES_DIR/environment.sh"
cd "$PROJECT_ROOT"
: "${SOURCE_SNAPSHOT_SHA:?dispatcher must export SOURCE_SNAPSHOT_SHA}"
: "${MODEL_REVISION:?dispatcher must export MODEL_REVISION}"
CURRENT_SOURCE_SHA=$("$PYTHON" experiments/baselines/capture_provenance.py --source-sha)
[[ "$CURRENT_SOURCE_SHA" == "$SOURCE_SNAPSHOT_SHA" ]] || {
  echo "Experiment source snapshot changed after dispatch; refusing mixed-protocol resume." >&2
  exit 2
}
mkdir -p "$ATTEMPT_DIR"
[[ ! -e "$ATTEMPT_DIR/DONE" ]] || exit 0
RUN_ARGS=()
if [[ "${RESUME:-0}" == 1 ]]; then RUN_ARGS+=(--resume); fi
if (( ${QUERY_LIMIT:-0} > 0 )); then RUN_ARGS+=(--max-queries "$QUERY_LIMIT"); fi

"$PYTHON" experiments/baselines/run_tourrank_model_matched.py \
  --dataset "$DATASET" --local-run "$LOCAL_RUN" \
  --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --cache-dir "${HF_HOME:-$PROJECT_ROOT/.cache/hf}" \
  --output-dir "$ATTEMPT_DIR" --tournaments "$TOURNAMENTS" \
  "${RUN_ARGS[@]}" \
  2>&1 | tee "$ATTEMPT_DIR/tourrank.log"

"$PYTHON" experiments/baselines/finalize_official_attempt.py \
  --attempt-dir "$ATTEMPT_DIR" --condition-dir "$(dirname "$ATTEMPT_DIR")" \
  --run-file tourrank.txt --telemetry-file tourrank_telemetry.jsonl \
  --summary-file tourrank_summary.json \
  --input-run "$LOCAL_RUN" --dataset "$DATASET" \
  --output-objective top10 --output-depth 10 \
  --expected-category model_matched_adapted \
  --query-limit "${QUERY_LIMIT:-0}" \
  --expected-calls-per-query "$((13 * TOURNAMENTS))"
