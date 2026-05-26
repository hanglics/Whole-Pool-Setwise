#!/usr/bin/env bash
# Submit one EMNLP main-matrix cell, optionally across all six pool sizes.
#
# Reuses experiments/run_*.sh as the SLURM job scripts so module loads, conda
# env activation, and the cd into the project root all happen inside the job
# exactly as they do for the IDEA_007 dispatcher (submit_max_context_jobs.sh).
# Per-cell behavior is steered via sbatch --export env vars:
#
#   CONDA_ENV          ranker_env (Qwen3 + pyserini) for Qwen3-* models;
#                      qwen35_env for Qwen3.5 / Llama-3.1 / Ministral-3.
#   ANALYSIS_LOG_DIR   directory for *_comparisons.jsonl, set to OUTPUT_DIR so
#                      different (model, dataset, pool) cells don't collide on
#                      the launcher's default basename-derived path.
#   LLMRANKERS_OVERWRITE_RESULTS
#                      1 when --overwrite was passed; launchers use it for a
#                      final defensive skip check before run.py.

set -euo pipefail

METHOD=""
MODEL=""
DATASET=""
POOL_SIZE="50"
TAG=""
DRY_RUN=0
MAX_JOBS=100
OUTPUT_ROOT=""
TIME_LIMIT=""  # empty = method×dataset-aware default resolved after arg parsing
SHUFFLE=0
REVERSE=0
FULL_WINDOW=0
OVERWRITE=0
ALLOW_PARSE_FAILURE_BM25_FALLBACK=0

usage() {
  cat <<'USAGE'
Usage: submit_emnlp_jobs.sh --method METHOD --model HF_ID --dataset DATASET --tag TAG [options]

Options:
  --method METHOD     One of topdown_bubblesort, topdown_heapsort,
                      bottomup_bubblesort, bottomup_heapsort,
                      maxcontext_topdown, maxcontext_bottomup,
                      maxcontext_dualend.
  --model HF_ID       HuggingFace model id. Family-mapped to a conda env:
                        Qwen/Qwen3-*                 -> ranker_env
                        Qwen/Qwen3.5-*               -> qwen35_env
                        meta-llama/Meta-Llama-3.1-*  -> qwen35_env
                        mistralai/Ministral-3-*      -> qwen35_env
                        anything else                -> ranker_env (fallback)
  --dataset DATASET   dl19, dl20, beir-dbpedia, beir-nfcorpus, beir-scifact,
                      beir-trec-covid, beir-touche2020, or beir-fiqa.
  --pool-size N|all   Pool size (default: 50). Use "all" for 10,20,30,40,50,100.
  --tag TAG           Main-matrix tag under results/emnlp/main/.
  --output-root DIR   Override output root; default is results/emnlp/main/TAG.
  --max-jobs N        Refuse to submit if expanded job count exceeds N
                      (default: 100).
  --time-limit HMS    SLURM --time directive override. If unset, a method×dataset
                      default is selected after arg parsing (see _resolve_time_limit
                      below). Bubblesort is O(N²) and BEIR datasets have 50–650 queries
                      vs. DL19/20's 43–54, so the defaults scale accordingly.
  --shuffle           MaxContext-only: per-round shuffle of the remaining pool.
  --reverse           MaxContext-only: per-round reverse of the remaining pool.
  --full-window       Non-MaxContext setwise only: use WS=PS
                      (NUM_CHILD=POOL_SIZE-1, K=POOL_SIZE) instead of
                      the default NUM_CHILD=2, K=10. Historical Phase B1a
                      support; current EMNLP scope skips B1a.
  --allow-parse-failure-bm25-fallback
                      MaxContext-only: enable per-query BM25 fallback when LLM
                      label parsing fails (off by default — strict-raise is
                      preserved so smoke runs surface new parse-failure modes).
                      Recommended for Phase B/C/F main-matrix dispatch.
  --dry-run           Print sbatch commands instead of submitting.
  --overwrite         Re-submit even if the target .txt result file already
                      exists and is non-empty. Default: skip with an [INFO]
                      message.
  -h | --help         Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) [[ $# -ge 2 ]] || { echo "Error: --method requires a value" >&2; exit 2; }; METHOD="$2"; shift 2 ;;
    --model) [[ $# -ge 2 ]] || { echo "Error: --model requires a value" >&2; exit 2; }; MODEL="$2"; shift 2 ;;
    --dataset) [[ $# -ge 2 ]] || { echo "Error: --dataset requires a value" >&2; exit 2; }; DATASET="$2"; shift 2 ;;
    --pool-size) [[ $# -ge 2 ]] || { echo "Error: --pool-size requires a value" >&2; exit 2; }; POOL_SIZE="$2"; shift 2 ;;
    --tag) [[ $# -ge 2 ]] || { echo "Error: --tag requires a value" >&2; exit 2; }; TAG="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || { echo "Error: --output-root requires a value" >&2; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
    --max-jobs) [[ $# -ge 2 ]] || { echo "Error: --max-jobs requires a value" >&2; exit 2; }; MAX_JOBS="$2"; shift 2 ;;
    --time-limit) [[ $# -ge 2 ]] || { echo "Error: --time-limit requires a value" >&2; exit 2; }; TIME_LIMIT="$2"; shift 2 ;;
    --shuffle) SHUFFLE=1; shift ;;
    --reverse) REVERSE=1; shift ;;
    --full-window) FULL_WINDOW=1; shift ;;
    --allow-parse-failure-bm25-fallback) ALLOW_PARSE_FAILURE_BM25_FALLBACK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$METHOD" ]] || { echo "Error: --method is required" >&2; usage >&2; exit 2; }
[[ -n "$MODEL" ]] || { echo "Error: --model is required" >&2; usage >&2; exit 2; }
[[ -n "$DATASET" ]] || { echo "Error: --dataset is required" >&2; usage >&2; exit 2; }
[[ -n "$TAG" || -n "$OUTPUT_ROOT" ]] || { echo "Error: --tag is required unless --output-root is set" >&2; usage >&2; exit 2; }

if [[ "$SHUFFLE" -eq 1 && "$REVERSE" -eq 1 ]]; then
  echo "Error: --shuffle and --reverse are mutually exclusive" >&2
  exit 2
fi

case "$DATASET" in
  dl19)
    DATASET_PATH="msmarco-passage/trec-dl-2019/judged"
    BM25_RUN="runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"
    DATASET_TAG="dl19"
    ;;
  dl20)
    DATASET_PATH="msmarco-passage/trec-dl-2020/judged"
    BM25_RUN="runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt"
    DATASET_TAG="dl20"
    ;;
  beir-dbpedia)
    DATASET_PATH="beir/dbpedia-entity/test"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.dbpedia-entity.txt"
    DATASET_TAG="beir-dbpedia"
    ;;
  beir-nfcorpus)
    DATASET_PATH="beir/nfcorpus/test"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.nfcorpus.txt"
    DATASET_TAG="beir-nfcorpus"
    ;;
  beir-scifact)
    DATASET_PATH="beir/scifact/test"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.scifact.txt"
    DATASET_TAG="beir-scifact"
    ;;
  beir-trec-covid)
    DATASET_PATH="beir/trec-covid"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.trec-covid.txt"
    DATASET_TAG="beir-trec-covid"
    ;;
  beir-touche2020)
    DATASET_PATH="beir/webis-touche2020/v2"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.webis-touche2020.txt"
    DATASET_TAG="beir-touche2020"
    ;;
  beir-fiqa)
    DATASET_PATH="beir/fiqa/test"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.fiqa.txt"
    DATASET_TAG="beir-fiqa"
    ;;
  *) echo "Error: unsupported --dataset '$DATASET'" >&2; exit 2 ;;
esac

# Map method -> launcher script + k policy.
# K_MODE=="pool" : MaxContext direction. Launcher takes
#                  (MODEL DS RUN_PATH OUTPUT_DIR DEVICE SCORING POOL_SIZE PASSAGE_LENGTH);
#                  ranker overrides num_child to pool_size-1 internally.
# K_MODE=="fixed": standard setwise. Bigram launcher takes
#                  (MODEL DS RUN_PATH OUTPUT_DIR DEVICE SCORING NUM_CHILD K HITS PASSAGE_LENGTH METHOD);
#                  EMNLP standard methods use num_child=2 (Setwise WS=3), k=10, hits=pool_size.
case "$METHOD" in
  topdown_heapsort)    LAUNCHER="experiments/run_topdown_bigram.sh";    K_MODE="fixed"; SETWISE_METHOD="heapsort"   ;;
  topdown_bubblesort)  LAUNCHER="experiments/run_topdown_bigram.sh";    K_MODE="fixed"; SETWISE_METHOD="bubblesort" ;;
  bottomup_heapsort)   LAUNCHER="experiments/run_bottomup_bigram.sh";   K_MODE="fixed"; SETWISE_METHOD="heapsort"   ;;
  bottomup_bubblesort) LAUNCHER="experiments/run_bottomup_bigram.sh";   K_MODE="fixed"; SETWISE_METHOD="bubblesort" ;;
  maxcontext_topdown)  LAUNCHER="experiments/run_maxcontext_topdown.sh";  K_MODE="pool"  ;;
  maxcontext_bottomup) LAUNCHER="experiments/run_maxcontext_bottomup.sh"; K_MODE="pool"  ;;
  maxcontext_dualend)  LAUNCHER="experiments/run_maxcontext_dualend.sh";  K_MODE="pool"  ;;
  *) echo "Error: unsupported --method '$METHOD'" >&2; exit 2 ;;
esac

if [[ "$FULL_WINDOW" -eq 1 && "$K_MODE" == "pool" ]]; then
  echo "Error: --full-window is supported only for non-MaxContext setwise methods" >&2
  echo "       (MaxContext methods already use the full pool window)." >&2
  exit 2
fi

if [[ "$FULL_WINDOW" -eq 1 && ( "$SHUFFLE" -eq 1 || "$REVERSE" -eq 1 ) ]]; then
  echo "Error: --full-window is incompatible with --shuffle/--reverse" >&2
  echo "       (position-bias conditions are MaxContext-only)." >&2
  exit 2
fi

if [[ "$K_MODE" != "pool" && ( "$SHUFFLE" -eq 1 || "$REVERSE" -eq 1 ) ]]; then
  echo "Error: --shuffle/--reverse are supported only for MaxContext methods" >&2
  exit 2
fi

if [[ "$TAG" == "phase_b2_beir" || "$OUTPUT_ROOT" == *"phase_b2_beir"* ]]; then
  case "$DATASET_TAG" in
    beir-*) ;;
    *)
      echo "Error: tag phase_b2_beir is restricted to BEIR datasets" >&2
      exit 2
      ;;
  esac
  case "$METHOD" in
    maxcontext_topdown|maxcontext_dualend) ;;
    *)
      echo "Error: current Phase B2 scope allows only maxcontext_topdown and maxcontext_dualend" >&2
      exit 2
      ;;
  esac
  if [[ "$POOL_SIZE" != "100" ]]; then
    echo "Error: current Phase B2 scope requires --pool-size 100" >&2
    exit 2
  fi
  if [[ "$FULL_WINDOW" -eq 1 || "$SHUFFLE" -eq 1 || "$REVERSE" -eq 1 ]]; then
    echo "Error: phase_b2_beir does not allow --full-window, --shuffle, or --reverse" >&2
    exit 2
  fi
fi

# Resolve conda env from model family.
MODEL_BASE="${MODEL##*/}"
case "$MODEL_BASE" in
  Qwen3.5-*|Meta-Llama-3.1-*|Ministral-3-*)
    CONDA_ENV_PATH="${QWEN35_CONDA_ENV:-qwen35_env}" ;;
  *)
    CONDA_ENV_PATH="${RANKER_CONDA_ENV:-ranker_env}" ;;
esac

if [[ "$POOL_SIZE" == "all" ]]; then
  POOL_SIZES=(10 20 30 40 50 100)
elif [[ "$POOL_SIZE" =~ ^[0-9]+$ ]]; then
  POOL_SIZES=("$POOL_SIZE")
else
  echo "Error: --pool-size must be an integer or all" >&2
  exit 2
fi

if [[ "$FULL_WINDOW" -eq 1 ]]; then
  for N in "${POOL_SIZES[@]}"; do
    if (( N < 3 )); then
      echo "Error: --full-window requires --pool-size >= 3 (got $N)" >&2
      echo "       (NUM_CHILD=POOL_SIZE-1 would be < 2)." >&2
      exit 2
    fi
  done
fi

if [[ "${#POOL_SIZES[@]}" -gt "$MAX_JOBS" ]]; then
  echo "Would submit ${#POOL_SIZES[@]} jobs; --max-jobs cap is ${MAX_JOBS}" >&2
  exit 2
fi

MODEL_TAG="$(printf '%s' "${MODEL##*/}" | tr './' '-' | tr '[:upper:]' '[:lower:]')"
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="results/emnlp/main/${TAG}"
fi

CONDITION_SUFFIX=""
CONDITION_EXPORT=""
if [[ "$SHUFFLE" -eq 1 ]]; then
  CONDITION_SUFFIX="_shuffle"
  CONDITION_EXPORT=",SHUFFLE=1"
elif [[ "$REVERSE" -eq 1 ]]; then
  CONDITION_SUFFIX="_reverse"
  CONDITION_EXPORT=",REVERSE=1"
fi

# MaxContext-only: BM25 fallback flag is propagated to the launcher via env.
# The launcher reads ALLOW_PARSE_FAILURE_BM25_FALLBACK and passes
# --allow_parse_failure_bm25_fallback to run.py if set non-zero.
FALLBACK_EXPORT=""
if [[ "$ALLOW_PARSE_FAILURE_BM25_FALLBACK" -eq 1 ]]; then
  if [[ "$K_MODE" != "pool" ]]; then
    echo "Error: --allow-parse-failure-bm25-fallback is supported only for MaxContext methods" >&2
    exit 2
  fi
  FALLBACK_EXPORT=",ALLOW_PARSE_FAILURE_BM25_FALLBACK=1"
fi

# Method × dataset time-limit defaults. CLI --time-limit always wins;
# this resolver only fires when --time-limit is empty.
#
# Calibration anchors (observed on H100):
#   * DL19 + bubblesort       ~14h (43 queries × O(N²) calls)
#   * DL19 + heapsort/MaxCtx  <10h
# BEIR query counts: touche2020=49, trec-covid=50, nfcorpus=323, scifact=300,
# dbpedia=400, fiqa=648. Bubblesort cost scales linearly with #queries.
# Defaults are headroom-padded; pass --time-limit to override.
if [[ -z "$TIME_LIMIT" ]]; then
  case "${METHOD}:${DATASET}" in
    # --- Bubblesort × large BEIR (300+ queries): multi-day jobs ---
    topdown_bubblesort:beir-fiqa|bottomup_bubblesort:beir-fiqa)               TIME_LIMIT="168:00:00" ;;
    topdown_bubblesort:beir-dbpedia|bottomup_bubblesort:beir-dbpedia)         TIME_LIMIT="120:00:00" ;;
    topdown_bubblesort:beir-nfcorpus|bottomup_bubblesort:beir-nfcorpus)       TIME_LIMIT="96:00:00"  ;;
    topdown_bubblesort:beir-scifact|bottomup_bubblesort:beir-scifact)         TIME_LIMIT="96:00:00"  ;;
    # --- Bubblesort × small BEIR (~50 queries): similar scale to DL19/20 ---
    topdown_bubblesort:beir-touche2020|bottomup_bubblesort:beir-touche2020)   TIME_LIMIT="24:00:00"  ;;
    topdown_bubblesort:beir-trec-covid|bottomup_bubblesort:beir-trec-covid)   TIME_LIMIT="24:00:00"  ;;
    # --- Bubblesort × DL19/DL20 (43–54 queries): observed ~14h, pad to 24h ---
    topdown_bubblesort:dl19|topdown_bubblesort:dl20)                          TIME_LIMIT="24:00:00"  ;;
    bottomup_bubblesort:dl19|bottomup_bubblesort:dl20)                        TIME_LIMIT="24:00:00"  ;;
    # --- Non-bubblesort × large BEIR ---
    *:beir-fiqa)                                                              TIME_LIMIT="60:00:00"  ;;
    *:beir-dbpedia)                                                           TIME_LIMIT="48:00:00"  ;;
    *:beir-nfcorpus|*:beir-scifact)                                           TIME_LIMIT="36:00:00"  ;;
    # --- Non-bubblesort × small BEIR ---
    *:beir-touche2020|*:beir-trec-covid)                                      TIME_LIMIT="12:00:00"  ;;
    # --- Default: DL19/DL20 + non-bubblesort ---
    *)                                                                        TIME_LIMIT="10:00:00"  ;;
  esac
fi

JOB_COUNT=0
SKIP_COUNT=0
for N in "${POOL_SIZES[@]}"; do
  printf -v POOL_TAG "pool%02d" "$N"
  POOL_TAG="${POOL_TAG}${CONDITION_SUFFIX}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_TAG}/${DATASET_TAG}/${METHOD}/${POOL_TAG}"

  EXPECTED_TXT="${OUTPUT_DIR}/${METHOD}.txt"
  if [[ -s "$EXPECTED_TXT" && "$OVERWRITE" -eq 0 ]]; then
    echo "[INFO] Result file already exists: $EXPECTED_TXT" >&2
    echo "[INFO] Skipping submission. If overwrite, add \"--overwrite\"." >&2
    SKIP_COUNT=$((SKIP_COUNT + 1))
    continue
  fi

  if [[ "$K_MODE" == "pool" ]]; then
    LAUNCHER_ARGS=("$MODEL" "$DATASET_PATH" "$BM25_RUN" "$OUTPUT_DIR" cuda generation "$N" 512)
  elif [[ "$FULL_WINDOW" -eq 1 ]]; then
    # Phase B1a full-pool window: NUM_CHILD=POOL_SIZE-1, K=POOL_SIZE, HITS=POOL_SIZE.
    # Pattern matches submit_max_context_jobs.sh Blocks 3/4/10/11 (WS=PS).
    # Launcher auto-enables --character_scheme bigrams_aa_zz when NUM_CHILD >= 23.
    LAUNCHER_ARGS=("$MODEL" "$DATASET_PATH" "$BM25_RUN" "$OUTPUT_DIR" cuda generation "$((N - 1))" "$N" "$N" 512 "$SETWISE_METHOD")
  else
    # EMNLP standard methods: NUM_CHILD=2 (Setwise WS=3), K=10, HITS=N=pool_size, PL=512, METHOD=heap/bubble.
    LAUNCHER_ARGS=("$MODEL" "$DATASET_PATH" "$BM25_RUN" "$OUTPUT_DIR" cuda generation 2 10 "$N" 512 "$SETWISE_METHOD")
  fi

  EXPORT_VARS="ALL,CONDA_ENV=${CONDA_ENV_PATH},ANALYSIS_LOG_DIR=${OUTPUT_DIR}${CONDITION_EXPORT}${FALLBACK_EXPORT},LLMRANKERS_OVERWRITE_RESULTS=${OVERWRITE}"
  if [[ "$FULL_WINDOW" -eq 1 ]]; then
    JOB_NAME="emnlp-fw-${METHOD}-${N}"
  else
    JOB_NAME="emnlp-${METHOD}-${N}"
  fi

  JOB_COUNT=$((JOB_COUNT + 1))
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'sbatch'
    printf ' --job-name=%s' "$JOB_NAME"
    printf ' --time=%s' "$TIME_LIMIT"
    printf ' --output=%s/slurm-%%j.out' "$OUTPUT_DIR"
    printf ' --export=%s' "$EXPORT_VARS"
    printf ' %s' "$LAUNCHER" "${LAUNCHER_ARGS[@]}"
    printf '\n'
  else
    mkdir -p "$OUTPUT_DIR"
    sbatch \
      --job-name="$JOB_NAME" \
      --time="$TIME_LIMIT" \
      --output="${OUTPUT_DIR}/slurm-%j.out" \
      --export="$EXPORT_VARS" \
      "$LAUNCHER" "${LAUNCHER_ARGS[@]}"
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: ${JOB_COUNT} job(s) would be submitted; ${SKIP_COUNT} skipped because result exists." >&2
else
  echo "Submitted ${JOB_COUNT} job(s); skipped ${SKIP_COUNT} existing result(s)." >&2
fi
