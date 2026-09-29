#!/usr/bin/env python3
"""Render the compact paper result pages from their checked-in CSVs."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
MODELS = {
    'qwen3-5-0-8b': 'Qwen3.5-0.8B', 'qwen3-5-2b': 'Qwen3.5-2B',
    'qwen3-5-4b': 'Qwen3.5-4B', 'qwen3-5-9b': 'Qwen3.5-9B',
    'qwen3-5-27b': 'Qwen3.5-27B', 'meta-llama-3-1-8b-instruct': 'Llama3.1-8B',
    'ministral-3-3b-instruct-2512': 'Ministral3-3B',
    'ministral-3-8b-instruct-2512': 'Ministral3-8B',
    'ministral-3-14b-instruct-2512': 'Ministral3-14B',
}
DATASETS = {'beir-dbpedia':'DBpedia', 'beir-fiqa':'FiQA', 'beir-nfcorpus':'NFCorpus',
            'beir-scifact':'SciFact', 'beir-touche2020':'Touché-2020', 'beir-trec-covid':'TREC-COVID'}


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
            *['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows], '']


def score(value):
    return f'{float(value):.4f}' if value else '—'


def baselines():
    rows = read_csv(DOCS/'data/baselines/summary.csv')
    lines = ['# Baseline comparisons', '', '[All detailed results](detailed-results.md) · [Reproduction](reproducing-paper.md)', '',
             'Pool 100 on DL19 (43 queries) and DL20 (54 queries), using the same three open models. TourRank-2 returns 10 documents; Liu zero-shot returns 100. WP-T and WP-DE stop after producing the top 10.', '',
             'These are model-matched adaptations. TourRank uses the official tournament logic and parser, serialized on one GPU; the Ministral runs include bounded format repair. Liu uses the official zero-shot prompt and permutation normalization. Calls and tokens include retries. Time is mean query inference time on an H100, excluding model loading.', '',
             '[CSV](data/baselines/summary.csv) · [Run provenance](data/baselines/provenance.json)', '']
    names = {'tourrank_2':'TourRank-2', 'liu_zero_shot':'Liu zero-shot', 'wp_t_top10':'WP-T top-10', 'wp_de_top10':'WP-DE top-10'}
    for model in ['Qwen/Qwen3.5-9B','meta-llama/Meta-Llama-3.1-8B-Instruct','mistralai/Ministral-3-8B-Instruct-2512']:
        lines += ['## '+model.split('/')[-1], '']
        cells=[]
        for dataset in ['dl19','dl20']:
            for method in names:
                r=next(r for r in rows if r['model']==model and r['dataset']==dataset and r['method']==method)
                cells.append([dataset.upper(),names[method],r['output_depth'],score(r['ndcg_cut_10']),score(r['ndcg_cut_100']),
                              f"{float(r['llm_calls_per_query']):.2f}",f"{float(r['prompt_tokens_per_query']):,.0f}",
                              f"{float(r['completion_tokens_per_query']):,.0f}",f"{float(r['seconds_per_query']):.2f}"])
        lines += table(['Dataset','Method','Depth','nDCG@10','nDCG@100','Calls/q','Input tokens/q','Output tokens/q','Seconds/q'],cells)
    (DOCS/'baseline-results.md').write_text('\n'.join(lines))


def beir():
    rows=read_csv(DOCS/'data/beir/beir_summary.csv')
    lines=['# BEIR results', '', '[All detailed results](detailed-results.md) · [Reproduction](reproducing-paper.md)', '',
           'Nine models, six datasets, WP-T and WP-DE, with a 100-document candidate cap. Scores use the frozen BM25 inputs; shorter pools retain all available candidates.', '',
           '[Scores CSV](data/beir/beir_summary.csv) · [Paired tests](data/beir/beir_significance.csv) · [Coverage and provenance](data/beir/README.md)', '',
           'n10 = nDCG@10; n100 = nDCG@100; MAP = MAP@100. MAP values and paired-test results retain the archived evaluations.', '']
    for dataset,name in DATASETS.items():
        bm=next(r for r in rows if r['dataset']==dataset and r['method']=='bm25' and not r['model'].startswith('__'))
        lines += ['## '+name,'',f"{bm['queries']} queries; candidate depths {bm['min_depth']}–{bm['max_depth']}. BM25: n10 **{score(bm['ndcg_cut_10'])}**, n100 **{score(bm['ndcg_cut_100'])}**, MAP **{score(bm['map_cut_100'])}**.",'']
        cells=[]
        for model,display in MODELS.items():
            a=next(r for r in rows if r['dataset']==dataset and r['model']==model and r['method']=='maxcontext_topdown')
            b=next(r for r in rows if r['dataset']==dataset and r['model']==model and r['method']=='maxcontext_dualend')
            cells.append([display,*[score(r[metric]) for metric in ['ndcg_cut_10','ndcg_cut_100','map_cut_100'] for r in [a,b]]])
        lines += table(['Model','WP-T n10','WP-DE n10','WP-T n100','WP-DE n100','WP-T MAP','WP-DE MAP'],cells)
    (DOCS/'beir-results.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    baselines()
    beir()
