#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

if [[ $# -ne 15 ]]; then
  echo "Usage: $0 MODEL DATASET_TAG DATASET_PATH RUN_PATH ATTEMPT_DIR METHOD_TAG DIRECTION SORT_METHOD POOL OUTPUT_DEPTH NUM_CHILD CHARACTER_SCHEME PROMPT_VARIANT LABEL_SCHEME ORDER" >&2
  exit 2
fi

MODEL=$1
DATASET_TAG=$2
DATASET_PATH=$3
RUN_PATH=$4
ATTEMPT_DIR=$5
METHOD_TAG=$6
DIRECTION=$7
SORT_METHOD=$8
POOL=$9
OUTPUT_DEPTH=${10}
NUM_CHILD=${11}
CHARACTER_SCHEME=${12}
PROMPT_VARIANT=${13}
LABEL_SCHEME=${14}
ORDER=${15}

[[ "$POOL" == 100 && "$OUTPUT_DEPTH" == 10 && "$SORT_METHOD" == selection && "$NUM_CHILD" == 99 && "$CHARACTER_SCHEME" == letters_a_w && "$PROMPT_VARIANT" == canonical && "$LABEL_SCHEME" == sequential && "$ORDER" == canonical ]] || {
  echo "This launcher implements the canonical pool-100, top-10 comparison." >&2
  exit 2
}
case "$METHOD_TAG:$DIRECTION" in
  wp_t_top10:maxcontext_topdown|wp_de_top10:maxcontext_dualend) ;;
  *) echo "Unsupported method/direction: $METHOD_TAG/$DIRECTION" >&2; exit 2 ;;
esac

SOURCE_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SOURCE_SCRIPT_DIR/../.." && pwd)}
BASELINES_DIR="$PROJECT_ROOT/experiments/baselines"
source "$BASELINES_DIR/environment.sh"
cd "$PROJECT_ROOT"
: "${MODEL_REVISION:?dispatcher must export immutable MODEL_REVISION}"
: "${SOURCE_SNAPSHOT_SHA:?dispatcher must export SOURCE_SNAPSHOT_SHA}"
CURRENT_SOURCE_SHA=$("$PYTHON" experiments/baselines/capture_provenance.py --source-sha)
if [[ "$CURRENT_SOURCE_SHA" != "$SOURCE_SNAPSHOT_SHA" ]]; then
  echo "Experiment source snapshot changed after dispatch; refusing mixed-protocol resume." >&2
  exit 2
fi

mkdir -p "$ATTEMPT_DIR"
if [[ -e "$ATTEMPT_DIR/DONE" ]]; then
  echo "Complete attempt already exists: $ATTEMPT_DIR" >&2
  exit 0
fi

METADATA="$ATTEMPT_DIR/experiment_metadata.json"
if [[ "${RESUME:-0}" != 1 || ! -s "$METADATA" ]]; then
  "$PYTHON" - "$METADATA" "$DATASET_TAG" "$METHOD_TAG" "$OUTPUT_DEPTH" \
  "$MODEL" "$MODEL_REVISION" "$RUN_PATH" "$DIRECTION" "$SORT_METHOD" \
  "$POOL" "$PROMPT_VARIANT" "$LABEL_SCHEME" "$ORDER" "$SOURCE_SNAPSHOT_SHA" \
  "$DATASET_PATH" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
(path, dataset, method, depth, model, model_revision, run_path, direction,
 sort_method, pool, prompt_variant, label_scheme, order, source_sha, dataset_path) = sys.argv[1:]
objective = "top10" if int(depth) == 10 else "full"
qrels = {"dl19": "dl19-passage", "dl20": "dl20-passage"}[dataset]
with open(run_path, "rb") as source:
    run_sha = hashlib.sha256(source.read()).hexdigest()
prompt_contract = {
    "direction": direction,
    "sort_method": sort_method,
    "pool": int(pool),
    "prompt_variant": prompt_variant,
    "label_scheme": label_scheme,
    "order": order,
    "source_snapshot_sha256": source_sha,
}
import ir_datasets
qrel_rows = sorted(
    (str(row.query_id), str(row.doc_id), int(row.relevance), str(getattr(row, "iteration", "0")))
    for row in ir_datasets.load(dataset_path).qrels_iter()
)
qrels_sha = hashlib.sha256(
    json.dumps(qrel_rows, separators=(",", ":")).encode()
).hexdigest()
with open(path, "w", encoding="utf-8") as stream:
    json.dump({
        "dataset": dataset,
        "method": method,
        "model_id": model,
        "model_revision": model_revision,
        "tokenizer_revision": model_revision,
        "output_objective": objective,
        "output_depth": int(depth),
        "first_stage_run": os.path.abspath(run_path),
        "first_stage_sha256": run_sha,
        "qrels_id": qrels,
        "qrels_sha256": qrels_sha,
        "ir_dataset_name": dataset_path,
        "source_snapshot_sha256": source_sha,
        "prompt_contract_sha256": hashlib.sha256(
            json.dumps(prompt_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_contract": prompt_contract,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
fi

RUN_ARGS=()
case "$ORDER" in
  canonical) ;;
  shuffle) RUN_ARGS+=(--shuffle) ;;
  reverse) RUN_ARGS+=(--reverse) ;;
  *) echo "Unknown order control: $ORDER" >&2; exit 2 ;;
esac
if [[ "${CAPTURE_RAW:-0}" == 1 ]]; then
  RUN_ARGS+=(--capture_raw_responses)
fi
if [[ "${RESUME:-0}" == 1 ]]; then
  RUN_ARGS+=(--resume)
fi
if (( ${QUERY_LIMIT:-0} > 0 )); then
  RUN_ARGS+=(--max_queries "$QUERY_LIMIT")
fi

if [[ "$DIRECTION" == maxcontext_* ]]; then
  SETWISE_K=$POOL
else
  SETWISE_K=$OUTPUT_DEPTH
fi

cmd=("$PYTHON" run.py
  run --model_name_or_path "$MODEL"
      --tokenizer_name_or_path "$MODEL"
      --ir_dataset_name "$DATASET_PATH"
      --run_path "$RUN_PATH"
      --save_path "$ATTEMPT_DIR/$METHOD_TAG.txt"
      --device cuda --scoring generation
      --model_revision "$MODEL_REVISION" --tokenizer_revision "$MODEL_REVISION"
      --hits "$POOL" --output_depth "$OUTPUT_DEPTH"
      --query_length 128 --passage_length 512
      --telemetry_path "$ATTEMPT_DIR/${METHOD_TAG}_telemetry.jsonl"
      --checkpoint_dir "$ATTEMPT_DIR/checkpoints"
      --experiment_metadata "$METADATA"
      --log_comparisons "$ATTEMPT_DIR/${METHOD_TAG}_comparisons.jsonl"
      "${RUN_ARGS[@]}"
  setwise --direction "$DIRECTION" --method "$SORT_METHOD"
      --num_child "$NUM_CHILD" --k "$SETWISE_K"
      --num_permutation 1 --character_scheme "$CHARACTER_SCHEME")

printf '[experiment] command:' | tee "$ATTEMPT_DIR/$METHOD_TAG.log"
printf ' %q' "${cmd[@]}" | tee -a "$ATTEMPT_DIR/$METHOD_TAG.log"
printf '\n' | tee -a "$ATTEMPT_DIR/$METHOD_TAG.log"
"${cmd[@]}" 2>&1 | tee -a "$ATTEMPT_DIR/$METHOD_TAG.log"

OBJECTIVE=full
if [[ "$OUTPUT_DEPTH" == 10 ]]; then OBJECTIVE=top10; fi
"$PYTHON" experiments/baselines/finalize_native_attempt.py \
  --attempt-dir "$ATTEMPT_DIR" \
  --condition-dir "$(dirname "$ATTEMPT_DIR")" \
  --method "$METHOD_TAG" --model "$MODEL" --dataset "$DATASET_TAG" \
  --output-objective "$OBJECTIVE" --output-depth "$OUTPUT_DEPTH" \
  --run-path "$RUN_PATH" --prompt-variant "$PROMPT_VARIANT" \
  --label-scheme "$LABEL_SCHEME" --seed 929 \
  --query-limit "${QUERY_LIMIT:-0}"
