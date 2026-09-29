# Reproducing the paper

[Overview](../README.md) · [Detailed results](detailed-results.md)

Run commands from the repository root. GPU launchers use SLURM; `--dry-run` prints commands without submitting jobs.

| Experiment | Scope | Results |
| --- | --- | --- |
| TREC DL | 9 models × 7 methods × 6 pools × DL19/DL20 | [Tables](detailed-results.md) |
| Baselines | TourRank-2 and Liu zero-shot × 3 models × DL19/DL20 | [Comparisons](baseline-results.md) |
| Matched depth | WP-T/WP-DE top-10 × 3 models × DL19/DL20 | [Comparisons](baseline-results.md) |
| BEIR | 9 models × 6 datasets × WP-T/WP-DE, cap 100 | [Per-dataset scores](beir-results.md) |
| Repeatability and order | DL19, pool 100, 3 representative models | [Controls](detailed-results.md#cost-and-diagnostic-controls) |

## Environment

The baseline and matched-depth runs used Python 3.10.20, PyTorch 2.10.0, Transformers 5.3.0, Accelerate 1.13.0, ir-datasets 0.5.6, and Pyserini 1.2.0. Install a PyTorch build compatible with your CUDA environment. Pyserini evaluation also requires a compatible Java runtime.

```bash
pip install -r requirements.txt
pip install -e .
```

Use `CONDA_ENV`, `RANKER_CONDA_ENV`, or `QWEN35_CONDA_ENV` for the baseline launchers, or leave them unset to use the active environment. `CONDA_SH`, `ANACONDA_MODULE`, and `JAVA_MODULE` are optional. See the [main launcher configuration](../README.md#setup) for the existing DL/BEIR scripts.

## Baselines and matched output depth

The three models are Qwen3.5-9B, Llama3.1-8B, and Ministral3-8B. Upstream code and model commits are pinned in `experiments/baselines/*.lock.json`.

```bash
python experiments/baselines/fetch_official_baselines.py
bash experiments/baselines/submit_jobs.sh --experiment tourrank --dry-run
bash experiments/baselines/submit_jobs.sh --experiment liu --dry-run
bash experiments/baselines/submit_jobs.sh --experiment matched_depth --max-jobs 12 --dry-run
```

Remove `--dry-run` to submit. After completion:

```bash
bash experiments/baselines/evaluate.sh --root results/paper
python experiments/baselines/collect_results.py \
  --root results/paper --output-dir results/paper/summary
```

TourRank-2 outputs 10 documents; Liu zero-shot outputs 100. The matched-depth WP-T/WP-DE runs output 10. These are adaptations to the paper's open models. Attempt manifests record prompt, model, input, source, and runtime provenance; telemetry counts all inference attempts, including repairs.

## BEIR

The six frozen BM25 inputs are included under `runs/bm25/`; verify them with `shasum -a 256 -c docs/data/beir/inputs.sha256`. Dataset loading uses `ir_datasets` and may download the corpora. The context probe samples passages from three model families; it does not certify every query.

```bash
python scripts/probe_beir_eval.py
python scripts/probe_beir_pool100_fit.py

MODELS=(Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B
        Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B
        meta-llama/Meta-Llama-3.1-8B-Instruct
        mistralai/Ministral-3-3B-Instruct-2512
        mistralai/Ministral-3-8B-Instruct-2512
        mistralai/Ministral-3-14B-Instruct-2512)
for MODEL in "${MODELS[@]}"; do
  for DATASET in beir-dbpedia beir-fiqa beir-nfcorpus beir-scifact beir-touche2020 beir-trec-covid; do
    for METHOD in maxcontext_topdown maxcontext_dualend; do
      bash submit_emnlp_jobs.sh --model "$MODEL" --dataset "$DATASET" \
        --method "$METHOD" --pool-size 100 --tag phase_b2_beir \
        --allow-parse-failure-bm25-fallback --dry-run
    done
  done
done
```

Remove `--dry-run` to submit. After completion, use the same loop with this evaluation command:

```bash
bash eval_emnlp_jobs.sh --model "$MODEL" --dataset "$DATASET" \
  --method "$METHOD" --pool-size 100 --tag phase_b2_beir --force
python analysis/beir_results.py \
  --root results/emnlp/main/phase_b2_beir --output-dir results/beir-summary
```

BEIR evaluation uses relevance threshold 1, per-query nDCG, and MAP@100. The audit checks query coverage, candidate membership, and complete `.log` or `.out` files, then produces summaries and paired tests. The published data retain archived MAP@100 and statistical results; see [data provenance](data/beir/README.md).

## Existing DL experiments and table rendering

Use the [DL matrix](../README.md#main-dl-matrix), [stability](../README.md#stability-runs), and input-order commands in the README. To regenerate the compact result pages from the bundled CSVs:

```bash
python scripts/render_paper_tables.py
```
