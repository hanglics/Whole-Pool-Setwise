# shellcheck shell=bash
# Source from a launcher after setting PROJECT_ROOT and BASELINES_DIR.
# An active Python environment is sufficient; module/conda setup is optional.
if [[ -n "${ANACONDA_MODULE:-}" ]]; then
  module load "$ANACONDA_MODULE"
fi
if [[ -n "${JAVA_MODULE:-}" ]]; then
  module load "$JAVA_MODULE"
fi
CONDA_ENV=${CONDA_ENV:-${QWEN35_CONDA_ENV:-${RANKER_CONDA_ENV:-}}}
if [[ -n "$CONDA_ENV" ]]; then
  if ! type conda >/dev/null 2>&1; then
    if [[ -n "${CONDA_SH:-}" ]]; then
      source "$CONDA_SH"
    elif [[ -n "${EBROOTANACONDA3:-}" ]]; then
      source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
    else
      echo "Set CONDA_SH to conda.sh, or activate your environment first." >&2
      return 2
    fi
  elif [[ $(type -t conda) != function ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
  fi
  source "$BASELINES_DIR/activate_conda.sh" "$CONDA_ENV"
  PYTHON="$CONDA_PREFIX/bin/python"
else
  PYTHON=${PYTHON:-$(command -v python3 || command -v python)}
fi
[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; return 127; }
export CACHE_ROOT=${CACHE_ROOT:-$PROJECT_ROOT/.cache}
export HF_HOME=${HF_HOME:-$CACHE_ROOT/hf}
export PYSERINI_CACHE=${PYSERINI_CACHE:-$CACHE_ROOT/pyserini}
export IR_DATASETS_HOME=${IR_DATASETS_HOME:-$CACHE_ROOT/ir_datasets}
