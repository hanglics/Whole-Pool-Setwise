import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "experiments" / "baselines"
DISPATCHER = BASELINES / "submit_jobs.sh"
CONDA_ACTIVATOR = BASELINES / "activate_conda.sh"
EXPERIMENT_RUNNERS = tuple(BASELINES / name for name in ("run_setwise.sh", "run_tourrank.sh", "run_liu.sh"))

def dispatcher_env():
    return {**os.environ, "PYTHON": sys.executable}

def test_experiment_conda_activation_tolerates_unset_java_library_path(tmp_path):
    env_path = tmp_path / "paper_env"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            r'''
set -euo pipefail
conda() {
  [[ "$1" == activate ]]
  : "$JAVA_LD_LIBRARY_PATH"
  export CONDA_PREFIX="$2"
}
source "$1" "$2"
case $- in *u*) ;; *) exit 91 ;; esac
[[ "$CONDA_PREFIX" == "$2" ]]
''',
            "bash",
            str(CONDA_ACTIVATOR),
            str(env_path),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_experiment_conda_activation_restores_nounset_after_failure(tmp_path):
    env_path = tmp_path / "paper_env"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            r'''
set -euo pipefail
conda() {
  : "$JAVA_LD_LIBRARY_PATH"
  return 17
}
if source "$1" "$2"; then
  exit 92
else
  status=$?
fi
[[ "$status" == 17 ]]
case $- in *u*) ;; *) exit 93 ;; esac
''',
            "bash",
            str(CONDA_ACTIVATOR),
            str(env_path),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_experiment_conda_activation_leaves_initially_disabled_nounset_disabled(tmp_path):
    env_path = tmp_path / "paper_env"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            r'''
set -eo pipefail
conda() {
  : "$JAVA_LD_LIBRARY_PATH"
  export CONDA_PREFIX="$2"
}
source "$1" "$2"
case $- in *u*) exit 94 ;; esac
[[ "$CONDA_PREFIX" == "$2" ]]
''',
            "bash",
            str(CONDA_ACTIVATOR),
            str(env_path),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_dispatcher_exports_project_root_for_slurm_spooled_scripts():
    cases = (
        ("tourrank", []),
        ("liu", []),
        ("matched_depth", ["--method", "wp_t_top10"]),
    )
    for phase, extra_args in cases:
        completed = subprocess.run(
            [
                str(DISPATCHER),
                "--experiment", phase,
                "--model", "Qwen/Qwen3.5-9B",
                "--dataset", "dl19",
                *extra_args,
                "--query-limit", "1",
                "--overwrite",
                "--dry-run",
                "--max-jobs", "1",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env=dispatcher_env(),
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert f"PROJECT_ROOT={ROOT}" in completed.stdout
        assert "/var/spool/slurmd" not in completed.stdout


def test_experiment_runners_never_resolve_helpers_beside_spooled_script():
    for runner in EXPERIMENT_RUNNERS:
        text = runner.read_text(encoding="utf-8")
        assert 'source "$SOURCE_SCRIPT_DIR/activate_conda.sh"' not in text
        assert 'source "$SCRIPT_DIR/activate_conda.sh"' not in text


def expanded_count(phase):
    completed = subprocess.run(
        [
            str(DISPATCHER), "--experiment", phase, "--overwrite",
            "--dry-run", "--max-jobs", "40",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dispatcher_env(),
    )
    assert completed.returncode == 0, (
        f"dispatcher failed for {phase} with exit {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    match = re.search(r"Expanded (\d+) unique job", completed.stdout)
    assert match, completed.stdout
    return int(match.group(1)), completed.stdout


def test_query_limited_smoke_is_isolated_from_production_attempts():
    completed = subprocess.run(
        [
            str(DISPATCHER),
            "--experiment", "matched_depth",
            "--model", "Qwen/Qwen3.5-9B",
            "--dataset", "dl19",
            "--method", "wp_de_top10",
            "--condition", "canonical",
            "--query-limit", "1",
            "--overwrite",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dispatcher_env(),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "/smoke-q1/" in completed.stdout
    assert "QUERY_LIMIT=1" in completed.stdout


def test_negative_query_limit_is_rejected():
    completed = subprocess.run(
        [str(DISPATCHER), "--experiment", "liu", "--query-limit", "-1", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dispatcher_env(),
    )
    assert completed.returncode == 2
    assert "--query-limit must be a non-negative integer" in completed.stderr


def test_dispatcher_does_not_require_platform_specific_shasum():
    assert "shasum" not in DISPATCHER.read_text()


def test_evaluator_ignores_tourrank_debug_sidecar(tmp_path):
    condition = (
        tmp_path
        / "tourrank"
        / "model"
        / "dl19"
        / "tourrank_2"
        / "model_matched_open"
    )
    attempt = condition / "abc-attempt"
    attempt.mkdir(parents=True)
    (condition / "LATEST").write_text("abc-attempt\n")
    (attempt / "DONE").write_text("ok\n")
    (attempt / "protocol_manifest.json").write_text(
        json.dumps(
            {"dataset": "dl19", "output_objective": "top10", "output_depth": 10}
        )
    )
    (attempt / "tourrank.txt").write_text("q Q0 d 1 -1 run\n")
    (attempt / "debug.txt").write_text("New Error:\nmalformed parser item\n")
    stale_debug_eval = attempt / "debug.eval"
    stale_debug_eval.write_text("not an evaluation\n")

    completed = subprocess.run(
        [
            str(ROOT / "experiments" / "baselines" / "evaluate.sh"),
            "--root",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        env=dispatcher_env(),
    )

    assert "tourrank.txt" in completed.stdout
    assert "debug.txt" not in completed.stdout
    assert f"[DRY-RUN] rm -f {stale_debug_eval}" in completed.stdout
    assert stale_debug_eval.is_file()


def test_paper_matrix_counts():
    for experiment, count in (("tourrank", 6), ("liu", 6), ("matched_depth", 12)):
        actual, output = expanded_count(experiment)
        assert actual == count
        assert "model_revisions" not in output  # pins are resolved before submission


def test_dry_run_never_creates_output_directories(tmp_path):
    output = tmp_path / "results"
    completed = subprocess.run(
        [str(DISPATCHER), "--experiment", "tourrank", "--dry-run", "--output-root", str(output)],
        cwd=ROOT, env=dispatcher_env(), text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not output.exists()


def test_source_inventory_is_complete_and_release_only():
    completed = subprocess.run(
        [sys.executable, str(BASELINES / "capture_provenance.py"), "--source-sha"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert re.fullmatch(r"[a-f0-9]{64}\n", completed.stdout)


def test_optional_environment_setup_uses_active_python(tmp_path):
    env = {key: value for key, value in os.environ.items() if key not in {
        "CONDA_ENV", "QWEN35_CONDA_ENV", "RANKER_CONDA_ENV", "ANACONDA_MODULE", "JAVA_MODULE"
    }}
    env.update(PROJECT_ROOT=str(ROOT), BASELINES_DIR=str(BASELINES), PYTHON=sys.executable)
    completed = subprocess.run(
        ["bash", "-eu", "-c", 'source "$BASELINES_DIR/environment.sh"; "$PYTHON" -c "print(123)"'],
        env=env, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "123"


def test_full_depth_evaluation_emits_ndcg_and_map(tmp_path):
    condition = tmp_path / "liu" / "model" / "dl20"
    attempt = condition / "abc-attempt"
    attempt.mkdir(parents=True)
    (condition / "LATEST").write_text("abc-attempt\n")
    (attempt / "DONE").write_text("ok\n")
    (attempt / "protocol_manifest.json").write_text(json.dumps({
        "dataset": "dl20", "output_objective": "full", "output_depth": 100, "run_file": "liu.txt"
    }))
    (attempt / "liu.txt").write_text("q Q0 d 1 -1 run\n")
    completed = subprocess.run(
        [str(BASELINES / "evaluate.sh"), "--root", str(tmp_path), "--dry-run"],
        cwd=ROOT, env=dispatcher_env(), text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert r"ndcg_cut.10\,100" in completed.stdout
    assert "map_cut.100" in completed.stdout
    assert "-l 2" in completed.stdout
