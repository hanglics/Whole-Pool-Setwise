"""BEIR coverage, execution-log selection and evaluation contracts."""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import beir_results as beir
from paired_significance import analyze

COMPLETE = ('Avg comparisons: 98.0\nAvg prompt tokens: 1000\n'
            'Avg completion tokens: 400\nAvg time per query: 12\n'
            'Total retries fired: 3\nAvg parse fallbacks: 0.02\n')

class BeirAuditTests(unittest.TestCase):
    def test_scheduler_output_substitutes_for_missing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'slurm-1.out'
            out.write_text(COMPLETE)
            self.assertEqual(beir.completion_marker_path(root / 'method.log', out), out)
            self.assertEqual(beir.parse_fallbacks(out)['retries_fired']['value'], 3)
            self.assertEqual(len(beir.sha256_file(out)), 64)

    def test_partial_output_is_not_a_completed_log(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'slurm-1.out'
            out.write_text('Avg time per query: 12\n')
            self.assertIsNone(beir.completion_marker_path(out))

    def test_complete_log_preferred_and_ambiguous_outputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log, first, second = [root / name for name in ('method.log', 'slurm-1.out', 'slurm-2.out')]
            for path in (log, first, second):
                path.write_text(COMPLETE)
            self.assertEqual(beir.completion_marker_path(log, first, second), log)
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                beir.completion_marker_path(first, second)

    def test_short_pools_preserve_exact_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run.txt'
            run.write_text('q1 Q0 b 1 2 tag\nq1 Q0 a 2 1 tag\nq2 Q0 c 1 1 tag\n')
            expected = {'q1': ['a', 'b'], 'q2': ['c']}
            self.assertEqual(beir.audit_run(run, expected)['errors'], [])
            run.write_text('q1 Q0 b 1 2 tag\nq1 Q0 b 2 1 tag\n')
            errors = beir.audit_run(run, expected)['errors']
            self.assertIn('duplicate_docids:q1', errors)
            self.assertIn('pool_membership_mismatch:q1', errors)
            self.assertIn('missing_qids:1', errors)

    def test_paired_tests_reject_mismatched_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b = [Path(directory)/name for name in ('a.eval','b.eval')]
            a.write_text('ndcg_cut_10 q1 0.5\n')
            b.write_text('ndcg_cut_10 q2 0.6\n')
            with self.assertRaisesRegex(ValueError, 'qid mismatch'):
                analyze([{'name':'pair','system_eval':a,'reference_eval':b,'metric':'ndcg_cut_10','direction':'higher'}],10,1,0.01)

    def test_eval_commands_keep_dataset_thresholds(self):
        for dataset,level in [('beir-fiqa','1'),('dl19','2')]:
            result = subprocess.run(['bash', str(ROOT/'eval_emnlp_jobs.sh'), '--method','maxcontext_topdown','--model','Qwen/Qwen3.5-9B','--dataset',dataset,'--pool-size','100','--tag','test','--dry-run'],capture_output=True,text=True,check=True)
            self.assertIn(f'-q -l {level}', result.stdout)
            self.assertEqual('map_cut.100' in result.stdout, dataset.startswith('beir-'))

if __name__ == '__main__':
    unittest.main()
