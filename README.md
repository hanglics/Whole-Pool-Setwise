# Whole-Pool Setwise Re-ranking

Code, experiment scripts, and detailed results for Whole-Pool Setwise re-ranking on TREC DL and BEIR, including model-matched TourRank and Liu baselines.

**[Detailed results and high-resolution tables](docs/detailed-results.md)** · **[Reproduce the paper](docs/reproducing-paper.md)**

It intentionally excludes paper drafts, planning/design notes, raw generated run outputs, logs, caches, conda environments, and local cluster paths. Curated appendix table images and their export manifest are included under `docs/assets/appendix-tables/`.

## Experiment Scope

The paper experiments are:

- Smoke gate: 3 representative models, 7 methods, DL19, pools 50 and 100.
- Main DL matrix: 9 models, 7 methods, DL19 and DL20, pools 10, 20, 30, 40, 50, and 100.
- Stability: 3 representative models, DL19, pool 100, 3 whole-pool methods, 5 repeated runs.
- Position-bias controls: the same 3 representative models and 3 whole-pool methods on DL19 pool 100 under reverse and fixed-seed shuffle conditions.

- BEIR: 9 models, 6 datasets, WP-T and WP-DE, candidate cap 100 (108 runs).
- Baselines: TourRank-2 and Liu zero-shot, 3 models, DL19/DL20 (12 runs).
- Matched output depth: WP-T/WP-DE top-10, the same 3 models and datasets (12 runs).

## Methods

The method names used by the scripts are:

| Script name | Paper abbreviation | Description |
|---|---|---|
| `topdown_bubblesort` | SW-TB | Setwise Windowed Top Bubblesort |
| `topdown_heapsort` | SW-TH | Setwise Windowed Top Heapsort |
| `bottomup_bubblesort` | SW-BB | Setwise Windowed Bottom Bubblesort |
| `bottomup_heapsort` | SW-BH | Setwise Windowed Bottom Heapsort |
| `maxcontext_topdown` | WP-T | Whole-Pool Setwise Top |
| `maxcontext_bottomup` | WP-B | Whole-Pool Setwise Bottom |
| `maxcontext_dualend` | WP-DE | Whole-Pool Setwise DualEnd |

## Setup

Create an environment with a compatible PyTorch/Transformers stack; the [reproduction guide](docs/reproducing-paper.md#environment) lists the tested versions. The Qwen3.5 and Ministral 3 checkpoints require a Transformers version that recognizes their model configs.

```bash
conda create -n wps python=3.10 -y
conda activate wps
pip install -r requirements.txt
pip install -e .
```

If your cluster uses modules, the SLURM launchers load `anaconda3/2023.09-0` by default. Override this when needed:

```bash
export ANACONDA_MODULE=anaconda3
```

The launchers are sanitized and use environment variables instead of local paths:

```bash
export PROJECT_ROOT="$PWD"
export RANKER_CONDA_ENV=wps
export QWEN35_CONDA_ENV=wps
export CACHE_ROOT="$PWD/.cache"
```

`RANKER_CONDA_ENV` and `QWEN35_CONDA_ENV` can be either conda environment names or absolute paths. If your cluster requires account, qos, partition, or node-exclusion options, edit the `#SBATCH` header lines in `experiments/*.sh` or pass site-specific options through your own wrapper.

For gated HuggingFace models such as Llama or Ministral, authenticate outside this repository:

```bash
huggingface-cli login
```

No tokens or credentials should be stored in this directory.

## Data Inputs

The DL19/DL20 and six BEIR BM25 first-stage runs are included. BEIR inputs have [recorded checksums](docs/data/beir/inputs.sha256).

DL inputs:

```text
runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt
runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt
```

They can also be regenerated with Pyserini:

```bash
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl19-passage \
  --output runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  --bm25 --k1 0.9 --b 0.4

python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl20-passage \
  --output runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt \
  --bm25 --k1 0.9 --b 0.4
```

## Smoke Gate

Run this first to verify the environment and the launch/evaluation path.

```bash
bash scripts/smoke_emnlp_models.sh --dry-run
bash scripts/smoke_emnlp_models.sh
```

After the jobs complete:

```bash
bash scripts/smoke_emnlp_models.sh --eval-only
bash scripts/smoke_emnlp_models.sh --verify-only
```

Smoke outputs are written under:

```text
results/emnlp/smoke/phase_a/
```

## Main DL Matrix

The main completed matrix is 9 models x 7 methods x 6 pool sizes x 2 datasets.

```bash
MODELS=(
  "Qwen/Qwen3.5-0.8B"
  "Qwen/Qwen3.5-2B"
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-27B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "mistralai/Ministral-3-3B-Instruct-2512"
  "mistralai/Ministral-3-8B-Instruct-2512"
  "mistralai/Ministral-3-14B-Instruct-2512"
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

POOL_SIZES=(10 20 30 40 50 100)
```

Submit the jobs. The loop below submits one model batch at a time; add your own queue-drain logic if your scheduler caps queued jobs.

```bash
for MODEL in "${MODELS[@]}"; do
  for DATASET in dl19 dl20; do
    for METHOD in "${METHODS[@]}"; do
      for POOL_SIZE in "${POOL_SIZES[@]}"; do
        EXTRA_ARGS=()
        if [[ "$METHOD" == maxcontext_* ]]; then
          EXTRA_ARGS+=(--allow-parse-failure-bm25-fallback)
        fi
        bash submit_emnlp_jobs.sh \
          --method "$METHOD" \
          --model "$MODEL" \
          --dataset "$DATASET" \
          --pool-size "$POOL_SIZE" \
          --tag phase_b1_dl \
          "${EXTRA_ARGS[@]}"
      done
    done
  done
done
```

Evaluate after all jobs complete:

```bash
for MODEL in "${MODELS[@]}"; do
  for DATASET in dl19 dl20; do
    for METHOD in "${METHODS[@]}"; do
      for POOL_SIZE in "${POOL_SIZES[@]}"; do
        bash eval_emnlp_jobs.sh \
          --method "$METHOD" \
          --model "$MODEL" \
          --dataset "$DATASET" \
          --pool-size "$POOL_SIZE" \
          --tag phase_b1_dl
      done
    done
  done
done
```

Main outputs are written under:

```text
results/emnlp/main/phase_b1_dl/{model_tag}/{dl19,dl20}/{method}/poolNN/
```

Each directory contains the TREC run file, run log, comparison diagnostics, and `.eval` file after evaluation.

Optional WP-DE pairwise significance checks can be run from the saved `.eval` files:

```bash
python analysis/wpde_significance.py \
  --root results/emnlp/main/phase_b1_dl \
  --output results/emnlp/analysis/wpde_significance.csv
```

## Stability Runs

The stability experiment uses DL19, pool 100, five repetitions, and only the three whole-pool methods.

```bash
STABILITY_MODELS=(
  "Qwen/Qwen3.5-9B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "mistralai/Ministral-3-8B-Instruct-2512"
)

for MODEL in "${STABILITY_MODELS[@]}"; do
  bash submit_emnlp_stability_jobs.sh \
    --model "$MODEL" \
    --dataset DL19 \
    --pool-sizes "100" \
    --reps 5
done
```

Evaluate:

```bash
for MODEL in "${STABILITY_MODELS[@]}"; do
  STABILITY_MODEL_TAG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
  for V in {1..5}; do
    bash eval_max_context_jobs.sh \
      --maxcontext-only \
      --pool-sizes "100" \
      --tag "emnlp_phase_c_required/${STABILITY_MODEL_TAG}-dl19/stability-test-runs/test_run_v${V}" \
      --model "$MODEL" \
      --dataset DL19
  done
done

python analysis/cross_model_stability.py
```

Analysis outputs are written under:

```text
results/emnlp/analysis/cross_model_stability/
```

## Position-Bias Controls

These runs reuse the forward BM25-order outputs from the main DL matrix and add reverse and fixed-seed shuffle conditions for DL19 pool 100.

```bash
POSITION_MODELS=(
  "Qwen/Qwen3.5-9B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "mistralai/Ministral-3-8B-Instruct-2512"
)

POSITION_METHODS=(
  "maxcontext_topdown"
  "maxcontext_bottomup"
  "maxcontext_dualend"
)

for MODEL in "${POSITION_MODELS[@]}"; do
  for METHOD in "${POSITION_METHODS[@]}"; do
    bash submit_emnlp_jobs.sh --reverse \
      --method "$METHOD" \
      --model "$MODEL" \
      --dataset dl19 \
      --pool-size 100 \
      --tag phase_f1_dl_reverse \
      --allow-parse-failure-bm25-fallback

    bash submit_emnlp_jobs.sh --shuffle \
      --method "$METHOD" \
      --model "$MODEL" \
      --dataset dl19 \
      --pool-size 100 \
      --tag phase_f1_dl_shuffle \
      --allow-parse-failure-bm25-fallback
  done
done
```

Evaluate:

```bash
for MODEL in "${POSITION_MODELS[@]}"; do
  for METHOD in "${POSITION_METHODS[@]}"; do
    bash eval_emnlp_jobs.sh --reverse \
      --method "$METHOD" \
      --model "$MODEL" \
      --dataset dl19 \
      --pool-size 100 \
      --tag phase_f1_dl_reverse

    bash eval_emnlp_jobs.sh --shuffle \
      --method "$METHOD" \
      --model "$MODEL" \
      --dataset dl19 \
      --pool-size 100 \
      --tag phase_f1_dl_shuffle
  done
done

python analysis/position_bias_emnlp.py \
  --main-root results/emnlp/main \
  --output-root results/emnlp/analysis/position_bias_emnlp
```

Analysis outputs are written under:

```text
results/emnlp/analysis/position_bias_emnlp/
```

## Single-Cell Debug Run

For debugging without the full submission wrapper:

```bash
bash experiments/run_maxcontext_dualend.sh \
  "Qwen/Qwen3.5-9B" \
  "msmarco-passage/trec-dl-2019/judged" \
  "runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt" \
  "results/debug/qwen3-5-9b/dl19/maxcontext_dualend/pool100" \
  cuda generation 100 512
```

Evaluate that run:

```bash
python -m pyserini.eval.trec_eval -q -l 2 \
  -m ndcg_cut.10,100 \
  dl19-passage \
  results/debug/qwen3-5-9b/dl19/maxcontext_dualend/pool100/maxcontext_dualend.txt
```

## Output Conventions

Model tags use the lower-cased basename of the HuggingFace ID with `/` and `.` converted to `-` for main EMNLP outputs. For example:

```text
Qwen/Qwen3.5-9B -> qwen3-5-9b
```

Pool directories use `poolNN` for canonical runs and `poolNN_reverse` / `poolNN_shuffle` for position-bias controls.

## Sanity Checks

Run lightweight tests:

```bash
python -m pytest tests
python scripts/check_maxcontext_invariants.py
```

Before sharing modified copies, rerun a local sensitive-pattern scan for site-specific paths and credentials.
