#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --job-name=bubg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EARLY_OUTPUT_DIR=${4:-"results/extended_setwise/qwen3-4b-dl19"}
EARLY_METHOD=${11:-"heapsort"}
EARLY_EXPECTED_TXT_PATH="${EARLY_OUTPUT_DIR}/bottomup_${EARLY_METHOD}.txt"
if [[ "$EARLY_EXPECTED_TXT_PATH" != /* ]]; then
  EARLY_EXPECTED_TXT_PATH="${PROJECT_ROOT}/${EARLY_EXPECTED_TXT_PATH}"
fi
if [[ -s "$EARLY_EXPECTED_TXT_PATH" && "${LLMRANKERS_OVERWRITE_RESULTS:-0}" != "1" ]]; then
  echo "[INFO] Result file already exists: $EARLY_EXPECTED_TXT_PATH" >&2
  echo "[INFO] Launcher exiting cleanly. To overwrite, set LLMRANKERS_OVERWRITE_RESULTS=1." >&2
  exit 0
fi

if command -v module >/dev/null 2>&1; then
  module load "${ANACONDA_MODULE:-anaconda3/2023.09-0}" || true
fi
if [[ -n "${EBROOTANACONDA3:-}" && -f "${EBROOTANACONDA3}/etc/profile.d/conda.sh" ]]; then
  source "${EBROOTANACONDA3}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "Error: conda is not available. Load an Anaconda/Miniconda module or put conda on PATH." >&2
  exit 2
fi
CONDA_ENV="${CONDA_ENV:-ranker_env}"
conda activate "$CONDA_ENV"
PYTHON="${CONDA_PREFIX:-$CONDA_ENV}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Error: selected CONDA_ENV python is not executable: $PYTHON" >&2
  echo "CONDA_ENV=$CONDA_ENV" >&2
  echo "CONDA_PREFIX=${CONDA_PREFIX:-}" >&2
  exit 2
fi
echo "[launcher] PROJECT_ROOT=$PROJECT_ROOT" >&2
echo "[launcher] CONDA_ENV=$CONDA_ENV" >&2
echo "[launcher] CONDA_PREFIX=${CONDA_PREFIX:-}" >&2
echo "[launcher] PYTHON=$PYTHON" >&2
"$PYTHON" -c 'import sys, ir_datasets; print("[launcher] sys.executable=" + sys.executable); print("[launcher] ir_datasets=" + ir_datasets.__file__)' >&2
cd "$PROJECT_ROOT"

MODEL=${1:-"Qwen/Qwen3-4B"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/extended_setwise/qwen3-4b-dl19"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
NUM_CHILD=${7:-3}
K=${8:-10}
HITS=${9:-100}
PASSAGE_LENGTH=${10:-512}
METHOD=${11:-"heapsort"}

mkdir -p ${OUTPUT_DIR}

if [[ -n "${ANALYSIS_LOG_DIR:-}" ]]; then
  ANALYSIS_DIR="${ANALYSIS_LOG_DIR}"
else
  ANALYSIS_DIR="results/analysis/position_bias/$(basename ${OUTPUT_DIR})"
fi
mkdir -p ${ANALYSIS_DIR}

EXPECTED_TXT_PATH="${OUTPUT_DIR}/bottomup_${METHOD}.txt"
if [[ -s "$EXPECTED_TXT_PATH" && "${LLMRANKERS_OVERWRITE_RESULTS:-0}" != "1" ]]; then
  echo "[INFO] Result file already exists: $EXPECTED_TXT_PATH" >&2
  echo "[INFO] Launcher exiting cleanly. To overwrite, set LLMRANKERS_OVERWRITE_RESULTS=1." >&2
  exit 0
fi

CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/hf}"
export PYSERINI_CACHE="${PYSERINI_CACHE:-${CACHE_ROOT}/pyserini}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-${CACHE_ROOT}/pyserini}"

CHARACTER_SCHEME_ARGS=()
if [ "${NUM_CHILD}" -ge 23 ]; then
    CHARACTER_SCHEME_ARGS=(--character_scheme bigrams_aa_zz)
fi

"${PYTHON}" run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/bottomup_${METHOD}.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/bottomup_${METHOD}_comparisons.jsonl \
    setwise --num_child ${NUM_CHILD} \
            --method ${METHOD} \
            --k ${K} \
            --direction bottomup \
            "${CHARACTER_SCHEME_ARGS[@]}" \
    2>&1 | tee ${OUTPUT_DIR}/bottomup_${METHOD}.log
