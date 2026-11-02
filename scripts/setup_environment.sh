#!/usr/bin/env bash
# Install homework dependencies and compile for the user's activated environment.
# Select a compatible CUDA PyTorch/torchvision pair before running this script.
set -euo pipefail

assignment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="${HW1_PYTHON:-python}"
export TMPDIR="${HW1_TMP_DIR:-${assignment_root}/.build_tmp}"
export PIP_CACHE_DIR="${TMPDIR}/pip_cache"
export TORCH_EXTENSIONS_DIR="${TMPDIR}/torch_extensions"
export MAX_JOBS="${MAX_JOBS:-4}"
dependency_index="${HW1_PIP_INDEX_URL:-https://pypi.org/simple}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

"$python_executable" -m pip install --index-url "$dependency_index" 'setuptools>=64' wheel
"$python_executable" "${assignment_root}/scripts/check_environment.py"
"$python_executable" -m pip install --index-url "$dependency_index" -r "${assignment_root}/requirements.txt"
"$python_executable" "${assignment_root}/scripts/check_environment.py" --check-runtime

cd "${assignment_root}/submodules/diff-gaussian-rasterization"
"$python_executable" setup.py clean --all
"$python_executable" -m pip install --no-build-isolation --no-deps --force-reinstall .
cd "$assignment_root"
"$python_executable" - <<'PY'
import diff_gaussian_rasterization as rasterizer
from diff_gaussian_rasterization import _C

print("Installed rasterizer:", rasterizer.__file__)
print("Compiled CUDA module:", _C.__file__)
PY

printf 'Environment ready. Run: %s %s/render.py\n' "$python_executable" "$assignment_root"
