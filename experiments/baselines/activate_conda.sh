# shellcheck shell=bash
# Source this helper after conda.sh: source activate_conda.sh ENV_PATH

experiment_activate_conda() {
  if [[ $# -ne 1 ]]; then
    echo "Usage: source activate_conda.sh ENV_PATH" >&2
    return 2
  fi

  local env_path=$1
  local restore_nounset=0
  local status=0
  case $- in
    *u*)
      restore_nounset=1
      set +u
      ;;
  esac

  # Conda activate.d hooks are not guaranteed to be compatible with `set -u`.
  # In particular, conda-forge OpenJDK reads JAVA_LD_LIBRARY_PATH before it has
  # necessarily been defined. Keep errexit/pipefail active and restore nounset
  # immediately after activation.
  if conda activate "$env_path"; then
    status=0
  else
    status=$?
  fi

  if (( restore_nounset )); then
    set -u
  fi
  return "$status"
}

experiment_activate_conda_status=0
experiment_activate_conda "$1" || experiment_activate_conda_status=$?
unset -f experiment_activate_conda
return "$experiment_activate_conda_status"
