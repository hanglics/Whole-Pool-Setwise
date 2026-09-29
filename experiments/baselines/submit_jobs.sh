#!/usr/bin/env bash
# Paper baseline and matched-depth experiment dispatcher.
set -euo pipefail

EXPERIMENT=""
MODEL_FILTER=""
DATASET_FILTER=""
METHOD_FILTER=""
CONDITION_FILTER=""
POOL_FILTER=""
OUTPUT_ROOT="results/paper"
DRY_RUN=0
RESUME=0
OVERWRITE=0
MAX_JOBS=8
QUERY_LIMIT=0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
cd "$PROJECT_ROOT"

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
CONDA_ENV_PATH=${CONDA_ENV:-${QWEN35_CONDA_ENV:-${RANKER_CONDA_ENV:-}}}

usage() {
  echo "Usage: $0 --experiment EXPERIMENT [--model MODEL] [--dataset dl19|dl20] [--method METHOD] [--condition CONDITION] [--pool-size N] [--output-root DIR] [--query-limit N] [--dry-run] [--resume] [--overwrite] [--max-jobs N]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment) EXPERIMENT=$2; shift 2 ;;
    --model) MODEL_FILTER=$2; shift 2 ;;
    --dataset) DATASET_FILTER=$2; shift 2 ;;
    --method) METHOD_FILTER=$2; shift 2 ;;
    --condition) CONDITION_FILTER=$2; shift 2 ;;
    --pool-size) POOL_FILTER=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume) RESUME=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --max-jobs) MAX_JOBS=$2; shift 2 ;;
    --query-limit) QUERY_LIMIT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$EXPERIMENT" ]] || { echo "--experiment is required" >&2; exit 2; }
[[ "$MAX_JOBS" =~ ^[0-9]+$ ]] || { echo "--max-jobs must be an integer" >&2; exit 2; }
[[ "$QUERY_LIMIT" =~ ^[0-9]+$ ]] || { echo "--query-limit must be a non-negative integer" >&2; exit 2; }

MODELS=(
  "Qwen/Qwen3.5-9B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "mistralai/Ministral-3-8B-Instruct-2512"
)
DATASETS=(dl19 dl20)
CELLS=()

allowed() {
  local model=$1 dataset=$2 method=$3 pool=$4
  [[ -z "$MODEL_FILTER" || "$model" == "$MODEL_FILTER" ]] || return 1
  [[ -z "$DATASET_FILTER" || "$dataset" == "$DATASET_FILTER" ]] || return 1
  [[ -z "$METHOD_FILTER" || "$method" == "$METHOD_FILTER" ]] || return 1
  [[ -z "$POOL_FILTER" || "$pool" == "$POOL_FILTER" ]] || return 1
}

add_cell() {
  local model=$1 dataset=$2 method=$3 pool=$4 condition=$5 direction=$6 sort=$7 num_child=$8 character=$9 prompt=${10} labels=${11} order=${12} capture=${13}
  if allowed "$model" "$dataset" "$method" "$pool" \
      && [[ -z "$CONDITION_FILTER" || "$condition" == "$CONDITION_FILTER" ]]; then
    CELLS+=("$model|$dataset|$method|$pool|$condition|$direction|$sort|$num_child|$character|$prompt|$labels|$order|$capture")
  fi
}

case "$EXPERIMENT" in
  tourrank)
    for model in "${MODELS[@]}"; do for dataset in "${DATASETS[@]}"; do
      add_cell "$model" "$dataset" "tourrank_2" 100 "model_matched_open" external external 0 none canonical sequential canonical 0
    done; done
    ;;
  liu)
    for model in "${MODELS[@]}"; do for dataset in "${DATASETS[@]}"; do
      add_cell "$model" "$dataset" "liu_zero_shot" 100 "model_matched_open" external external 0 none canonical sequential canonical 0
    done; done
    ;;
  matched_depth)
    for model in "${MODELS[@]}"; do for dataset in "${DATASETS[@]}"; do
      add_cell "$model" "$dataset" wp_t_top10 100 canonical maxcontext_topdown selection 99 letters_a_w canonical sequential canonical 0
      add_cell "$model" "$dataset" wp_de_top10 100 canonical maxcontext_dualend selection 99 letters_a_w canonical sequential canonical 0
    done; done
    ;;
  *) echo "Unsupported experiment: $EXPERIMENT" >&2; exit 2 ;;
esac

if (( ${#CELLS[@]} == 0 )); then
  echo "The requested filters are incompatible with experiment $EXPERIMENT; no jobs expanded." >&2
  exit 2
fi

echo "Expanded ${#CELLS[@]} unique job(s) for $EXPERIMENT."
if (( ${#CELLS[@]} > MAX_JOBS )); then
  echo "Refusing ${#CELLS[@]} jobs because --max-jobs=$MAX_JOBS" >&2
  exit 2
fi

OUTPUT_ROOT=$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_ROOT")
if (( QUERY_LIMIT > 0 )); then
  OUTPUT_ROOT="$OUTPUT_ROOT/smoke-q${QUERY_LIMIT}"
fi
SOURCE_SNAPSHOT_SHA=$("$PYTHON_BIN" experiments/baselines/capture_provenance.py --source-sha)
MODEL_LOCK="$SCRIPT_DIR/model_revisions.lock.json"

model_revision() {
  "$PYTHON_BIN" - "$MODEL_LOCK" "$1" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))
try:
    print(lock[sys.argv[2]])
except KeyError as exc:
    raise SystemExit(f"No immutable model revision for {sys.argv[2]}") from exc
PY
}


prepared=0
for cell in "${CELLS[@]}"; do
  IFS='|' read -r model dataset method pool condition direction sort num_child character prompt labels order capture <<< "$cell"
  model_base=${model##*/}
  model_tag=$(printf '%s' "$model_base" | tr './' '-' | tr '[:upper:]' '[:lower:]')
  dataset_path="msmarco-passage/trec-dl-2019/judged"
  input_run="runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"
  if [[ "$dataset" == dl20 ]]; then
    dataset_path="msmarco-passage/trec-dl-2020/judged"
    input_run="runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt"
  fi
  model_rev=$(model_revision "$model")
  if [[ -f "$input_run" ]]; then
    input_run=$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$input_run")
    input_sha=$("$PYTHON_BIN" -c 'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$input_run")
  elif [[ "$DRY_RUN" == 1 ]]; then
    input_sha="MISSING_DRY_RUN"
  else
    echo "Missing first-stage run: $input_run" >&2
    exit 2
  fi
  protocol_key="$EXPERIMENT|$model|$model_rev|$dataset|$method|$pool|$condition|$direction|$sort|$num_child|$character|$prompt|$labels|$order|$input_sha|$SOURCE_SNAPSHOT_SHA|query_limit=$QUERY_LIMIT"
  protocol_hash=$("$PYTHON_BIN" -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest()[:12])' "$protocol_key")
  condition_dir="$OUTPUT_ROOT/$EXPERIMENT/$model_tag/$dataset/$method/$condition"
  latest_file="$condition_dir/LATEST"
  if [[ -f "$latest_file" ]]; then
    latest=$(<"$latest_file")
    if [[ "$latest" == "${protocol_hash}-"* && -f "$condition_dir/$latest/DONE" && "$OVERWRITE" != 1 ]]; then
      echo "[skip] complete $condition_dir/$latest"
      continue
    fi
  fi
  incomplete=$(find "$condition_dir" -mindepth 1 -maxdepth 1 -type d ! -exec test -f '{}/DONE' \; -print 2>/dev/null | sort | tail -1 || true)
  if [[ -n "$incomplete" && "$OVERWRITE" != 1 ]]; then
    if [[ "$RESUME" != 1 ]]; then
      echo "Incomplete attempt exists; use --resume or --overwrite: $incomplete" >&2
      exit 2
    fi
    if [[ "$(basename "$incomplete")" != "${protocol_hash}-"* ]]; then
      echo "Incomplete attempt has a different protocol hash: $incomplete" >&2
      exit 2
    fi
  fi
  attempt=""
  if [[ "$RESUME" == 1 ]]; then
    attempt=$(find "$condition_dir" -maxdepth 1 -type d -name "${protocol_hash}-*" 2>/dev/null | sort | tail -1 || true)
  fi
  if [[ -z "$attempt" ]]; then
    attempt="$condition_dir/${protocol_hash}-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  if [[ "$DRY_RUN" != 1 ]]; then mkdir -p "$attempt"; fi

  cmd=(sbatch --chdir "$PROJECT_ROOT" --job-name "wps_${EXPERIMENT}_${dataset}_${method}" --output "$attempt/slurm-%j.out")
  if [[ "$EXPERIMENT" == tourrank ]]; then
    export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONDA_ENV=$CONDA_ENV_PATH,RESUME=$RESUME,QUERY_LIMIT=$QUERY_LIMIT,SOURCE_SNAPSHOT_SHA=$SOURCE_SNAPSHOT_SHA,MODEL_REVISION=$model_rev"
    cmd+=(--export "$export_spec" "$SCRIPT_DIR/run_tourrank.sh" "$model" "$dataset" "$input_run" "$attempt" 2)
  elif [[ "$EXPERIMENT" == liu ]]; then
    export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONDA_ENV=$CONDA_ENV_PATH,RESUME=$RESUME,QUERY_LIMIT=$QUERY_LIMIT,SOURCE_SNAPSHOT_SHA=$SOURCE_SNAPSHOT_SHA,MODEL_REVISION=$model_rev"
    cmd+=(--export "$export_spec" "$SCRIPT_DIR/run_liu.sh" "$model" "$dataset" "$input_run" "$attempt")
  else
    conda_env="$CONDA_ENV_PATH"
    export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONDA_ENV=$conda_env,CAPTURE_RAW=$capture,RESUME=$RESUME,QUERY_LIMIT=$QUERY_LIMIT,SOURCE_SNAPSHOT_SHA=$SOURCE_SNAPSHOT_SHA,MODEL_REVISION=$model_rev"
    cmd+=(--export "$export_spec" "$SCRIPT_DIR/run_setwise.sh"
      "$model" "$dataset" "$dataset_path" "$input_run" "$attempt" "$method"
      "$direction" "$sort" "$pool")
    if [[ "$EXPERIMENT" == matched_depth ]]; then
      cmd+=(10)
    else
      cmd+=("$pool")
    fi
    cmd+=("$num_child" "$character" "$prompt" "$labels" "$order")
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[DRY-RUN]'; printf ' %q' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
  fi
  prepared=$((prepared + 1))
done
echo "Prepared $prepared job(s)."
