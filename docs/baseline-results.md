# Baseline comparisons

[All detailed results](detailed-results.md) · [Reproduction](reproducing-paper.md)

Pool 100 on DL19 (43 queries) and DL20 (54 queries), using the same three open models. TourRank-2 returns 10 documents; Liu zero-shot returns 100. WP-T and WP-DE stop after producing the top 10.

These are model-matched adaptations. TourRank uses the official tournament logic and parser, serialized on one GPU; the Ministral runs include bounded format repair. Liu uses the official zero-shot prompt and permutation normalization. Calls and tokens include retries. Time is mean query inference time on an H100, excluding model loading.

[CSV](data/baselines/summary.csv) · [Run provenance](data/baselines/provenance.json)

## Qwen3.5-9B

| Dataset | Method | Depth | nDCG@10 | nDCG@100 | Calls/q | Input tokens/q | Output tokens/q | Seconds/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DL19 | TourRank-2 | 10 | 0.7376 | — | 26.00 | 42,913 | 761 | 23.79 |
| DL19 | Liu zero-shot | 100 | 0.7244 | 0.5723 | 1.00 | 8,958 | 491 | 24.15 |
| DL19 | WP-T top-10 | 10 | 0.7245 | — | 10.00 | 90,249 | 38 | 20.53 |
| DL19 | WP-DE top-10 | 10 | 0.7070 | — | 10.00 | 85,910 | 112 | 22.75 |
| DL20 | TourRank-2 | 10 | 0.6825 | — | 26.00 | 43,359 | 761 | 24.70 |
| DL20 | Liu zero-shot | 100 | 0.6772 | 0.5633 | 1.00 | 9,122 | 491 | 22.87 |
| DL20 | WP-T top-10 | 10 | 0.6842 | — | 10.00 | 92,657 | 60 | 22.16 |
| DL20 | WP-DE top-10 | 10 | 0.6744 | — | 10.00 | 88,085 | 112 | 23.22 |

## Meta-Llama-3.1-8B-Instruct

| Dataset | Method | Depth | nDCG@10 | nDCG@100 | Calls/q | Input tokens/q | Output tokens/q | Seconds/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DL19 | TourRank-2 | 10 | 0.6650 | — | 26.00 | 42,640 | 695 | 9.57 |
| DL19 | Liu zero-shot | 100 | 0.5739 | 0.5188 | 1.00 | 8,647 | 400 | 11.18 |
| DL19 | WP-T top-10 | 10 | 0.6502 | — | 10.00 | 87,287 | 30 | 6.03 |
| DL19 | WP-DE top-10 | 10 | 0.6275 | — | 10.00 | 83,351 | 99 | 7.34 |
| DL20 | TourRank-2 | 10 | 0.6631 | — | 26.00 | 43,031 | 696 | 9.95 |
| DL20 | Liu zero-shot | 100 | 0.5645 | 0.5118 | 1.00 | 8,773 | 400 | 11.69 |
| DL20 | WP-T top-10 | 10 | 0.5963 | — | 10.00 | 89,174 | 30 | 6.00 |
| DL20 | WP-DE top-10 | 10 | 0.5864 | — | 10.00 | 84,945 | 99 | 7.42 |

## Ministral-3-8B-Instruct-2512

| Dataset | Method | Depth | nDCG@10 | nDCG@100 | Calls/q | Input tokens/q | Output tokens/q | Seconds/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DL19 | TourRank-2 | 10 | 0.7116 | — | 26.00 | 40,385 | 775 | 32.26 |
| DL19 | Liu zero-shot | 100 | 0.7044 | 0.5662 | 1.00 | 8,895 | 491 | 32.28 |
| DL19 | WP-T top-10 | 10 | 0.7172 | — | 10.00 | 95,198 | 93 | 14.95 |
| DL19 | WP-DE top-10 | 10 | 0.7253 | — | 10.00 | 90,936 | 127 | 16.57 |
| DL20 | TourRank-2 | 10 | 0.7023 | — | 26.00 | 41,332 | 800 | 33.02 |
| DL20 | Liu zero-shot | 100 | 0.6840 | 0.5660 | 1.00 | 9,132 | 491 | 34.84 |
| DL20 | WP-T top-10 | 10 | 0.6811 | — | 10.00 | 98,044 | 77 | 14.10 |
| DL20 | WP-DE top-10 | 10 | 0.6772 | — | 10.00 | 93,617 | 126 | 16.58 |
