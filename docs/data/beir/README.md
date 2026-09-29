# BEIR data

Nine models × six datasets × two methods: 108 completed runs. The 100-document cap retains shorter input pools.

- `beir_summary.csv`: archived scores and equal-weight model averages.
- `beir_manifest.jsonl`: verified run coverage, execution logs and source checksums.
- `beir_significance.csv`: archived paired nDCG@10 tests, 10,000 samples, seed 929; Holm correction within each dataset family.
- `beir_fallbacks.csv`: parser diagnostics from execution logs.
- `bm25_manifest.jsonl`, `bm25_eval/`, `inputs.sha256`: frozen first-stage provenance.

Qwen3.5-27B/FiQA/WP-T uses the complete `slurm-25634351.out` as its log. MAP@100 retains the archived evaluation values; nDCG scores were checked against current outputs. The manifest distinguishes archived evaluation hashes from current nDCG-only evaluation hashes. Relative run paths identify the experiment layout; raw reranker outputs are not bundled.
