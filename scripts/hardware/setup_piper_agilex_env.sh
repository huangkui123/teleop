#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${XROBOT_ENV_NAME:-xrobotoolkit-native}"
SDK_DIR="${REPO_ROOT}/dependencies/XRoboToolkit-PC-Service-Pybind"

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda initialization script not found: ${CONDA_SH}" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"

if [[ ! -x "${HOME}/miniconda3/envs/${ENV_NAME}/bin/python" ]]; then
    conda create -n "${ENV_NAME}" python=3.10 pip -y
fi
conda activate "${ENV_NAME}"

# ROS Noetic exports Python 3.8 packages and libraries. They are ABI
# incompatible with the CPython 3.10 environment used by Placo and must not be
# inherited by pip, CMake, or the import smoke test below.
unset PYTHONPATH
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

python - <<'PY'
import platform
import sys

if platform.python_implementation() != "CPython" or sys.version_info < (3, 10):
    raise SystemExit(
        "PiPER teleoperation requires native CPython >= 3.10; got "
        f"{platform.python_implementation()} {platform.python_version()}"
    )
print(f"Using {sys.executable} ({platform.python_version()})")
PY

python -m pip install --upgrade pip
python -m pip install \
    "numpy>=2.2,<2.3" \
    "meshcat==0.3.2" \
    "placo==0.9.23" \
    "pybind11==3.0.4"

if [[ ! -f "${SDK_DIR}/include/PXREARobotSDK.h" ]] || \
   [[ ! -f "${SDK_DIR}/lib/libPXREARobotSDK.so" ]]; then
    echo "XR SDK header/library is missing under ${SDK_DIR}." >&2
    echo "Build it once with ${SDK_DIR}/setup_ubuntu.sh, then rerun this script." >&2
    exit 2
fi

PYBIND11_CMAKE_DIR="$(python -m pybind11 --cmakedir)"
export CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
python -m pip install "${SDK_DIR}" --no-build-isolation --force-reinstall
python -m pip install -e "${REPO_ROOT}" --no-deps

cd "${REPO_ROOT}"
python - <<'PY'
import meshcat
import numpy
import placo
import placo_utils
import xrobotoolkit_sdk
from xrobotoolkit_teleop.headless.piper import (
    create_dual_piper_joint_target_provider,
)

print("PiPER XR/IK imports passed.")
PY

echo "Environment ready. Activate it with: conda activate ${ENV_NAME}"
echo "Read-only robot check: python scripts/hardware/teleop_dual_piper_ros1.py --check"
