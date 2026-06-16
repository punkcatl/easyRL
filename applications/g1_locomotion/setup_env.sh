#!/bin/bash
# One-click environment setup for G1 Locomotion project
#
# Installs and configures everything needed to run the G1 humanoid RL training
# pipeline (Isaac Lab + rsl_rl + custom Teacher-Student distillation).
#
# Tested on: Ubuntu 22.04/24.04, NVIDIA Driver 580+, RTX A4000 16GB
# Requirements: conda, git, NVIDIA GPU driver
#
# Usage:
#   cd easyRL
#   bash applications/g1_locomotion/setup_env.sh
#
# What it does:
#   1. Creates/reuses conda env (env_isaaclab, Python 3.11)
#   2. Installs IsaacSim 5.1.0 + all extensions (isaacsim-rl)
#   3. Installs PyTorch 2.7.0+cu128
#   4. Clones & installs Isaac Lab 2.3.0 (with rsl_rl)
#   5. Installs isaaclab core + all missing Python deps
#   6. Patches Isaac Lab kit config for network-restricted envs
#   7. Sets up conda activate hooks (PYTHONPATH, ISAACLAB_PATH)
#   8. Runs full verification (import chain + GPU test)
#
# Pitfalls this script handles (learned the hard way):
#   - isaacsim pip package alone is NOT enough — need isaacsim-rl to pull extensions
#   - isaaclab core package must be pip install -e'd separately (not auto-installed)
#   - flatdict, prettytable, gymnasium, pkg_resources must be pinned/present
#   - setuptools >= 82 breaks pkg_resources (omni.kit.pipapi needs it)
#   - Isaac Sim registry sync to cloudfront can hang — kit file patched to skip community
#   - PYTHONPATH must include IsaacLab/source/* and easyRL root for custom configs

set -eo pipefail

#==============================================================================
# Configuration
#==============================================================================
CONDA_ENV_NAME="${CONDA_ENV_NAME:-env_isaaclab}"
PYTHON_VERSION="3.11"
ISAACLAB_VERSION="v2.3.0"
TORCH_VERSION="2.7.0"
TORCH_CUDA="cu128"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYRL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$(cd "${EASYRL_ROOT}/.." && pwd)/IsaacLab}"

ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
MAX_RETRIES=3
RETRY_DELAY=15

#==============================================================================
# Parse arguments
#==============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name) CONDA_ENV_NAME="$2"; shift 2 ;;
        --isaaclab-path) ISAACLAB_PATH="$(realpath "$2")"; shift 2 ;;
        --skip-torch) SKIP_TORCH=1; shift ;;
        --skip-isaacsim) SKIP_ISAACSIM=1; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env-name NAME        Conda env name (default: env_isaaclab)"
            echo "  --isaaclab-path PATH   Isaac Lab location (default: ../IsaacLab relative to easyRL)"
            echo "  --skip-torch           Skip PyTorch install (if already present)"
            echo "  --skip-isaacsim        Skip IsaacSim install (if already present)"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

#==============================================================================
# Helper functions
#==============================================================================
log() { echo -e "\n\033[1;32m[$(date '+%H:%M:%S')] $1\033[0m"; }
warn() { echo -e "\033[1;33m[WARN] $1\033[0m"; }
err() { echo -e "\033[1;31m[ERROR] $1\033[0m" >&2; exit 1; }

retry_pip() {
    local cmd="$*"
    for i in $(seq 1 $MAX_RETRIES); do
        if eval "$cmd"; then return 0; fi
        warn "pip attempt $i/$MAX_RETRIES failed. Retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    done
    err "pip install failed after $MAX_RETRIES attempts: $cmd"
}

check_nvidia() {
    command -v nvidia-smi >/dev/null 2>&1 || err "nvidia-smi not found. Install NVIDIA driver first."
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}

#==============================================================================
# Pre-flight checks
#==============================================================================
log "Pre-flight checks"

command -v conda >/dev/null 2>&1 || err "conda not found. Install Miniconda first."
command -v git >/dev/null 2>&1 || err "git not found."
echo "  GPU: $(check_nvidia)"
echo "  easyRL: ${EASYRL_ROOT}"
echo "  Isaac Lab: ${ISAACLAB_PATH}"
echo "  Conda env: ${CONDA_ENV_NAME}"

#==============================================================================
# Step 1: Clone Isaac Lab if missing
#==============================================================================
log "Step 1/8: Isaac Lab repository"

if [ -d "${ISAACLAB_PATH}" ]; then
    echo "  Already exists: ${ISAACLAB_PATH}"
    cd "${ISAACLAB_PATH}"
    echo "  Version: $(cat VERSION 2>/dev/null || echo 'unknown')"
else
    log "Cloning Isaac Lab ${ISAACLAB_VERSION}..."
    git clone --branch "${ISAACLAB_VERSION}" --depth 1 "${ISAACLAB_REPO}" "${ISAACLAB_PATH}"
fi

#==============================================================================
# Step 2: Create conda environment
#==============================================================================
log "Step 2/8: Conda environment '${CONDA_ENV_NAME}'"

eval "$(conda shell.bash hook)"

if conda env list | grep -qw "^${CONDA_ENV_NAME}"; then
    echo "  Already exists, reusing."
else
    conda create -n "${CONDA_ENV_NAME}" python=${PYTHON_VERSION} -y
fi
conda activate "${CONDA_ENV_NAME}"
echo "  Python: $(python --version)"

#==============================================================================
# Step 3: Install IsaacSim + extensions
#==============================================================================
log "Step 3/8: IsaacSim 5.1.0 + extensions"

if [ "${SKIP_ISAACSIM:-0}" = "1" ]; then
    echo "  Skipped (--skip-isaacsim)"
else
    ISAACSIM_VER=$(python -c "from importlib.metadata import version; print(version('isaacsim'))" 2>/dev/null || echo "none")
    if [ "$ISAACSIM_VER" = "5.1.0.0" ]; then
        echo "  isaacsim 5.1.0.0 already installed"
    else
        retry_pip "pip install isaacsim==5.1.0.0"
    fi

    # isaacsim-rl pulls all necessary extensions (robot, core, gui, etc.)
    # Without this, exts/ directory is empty and SimulationApp returns None
    ISAACSIM_RL_VER=$(python -c "from importlib.metadata import version; print(version('isaacsim-rl'))" 2>/dev/null || echo "none")
    if [ "$ISAACSIM_RL_VER" = "5.1.0.0" ]; then
        echo "  isaacsim-rl 5.1.0.0 already installed"
    else
        log "  Installing isaacsim-rl (this downloads ~2GB of extensions)..."
        retry_pip "pip install isaacsim-rl --extra-index-url https://pypi.nvidia.com"
    fi
fi

#==============================================================================
# Step 4: Install PyTorch
#==============================================================================
log "Step 4/8: PyTorch ${TORCH_VERSION}+${TORCH_CUDA}"

if [ "${SKIP_TORCH:-0}" = "1" ]; then
    echo "  Skipped (--skip-torch)"
else
    TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
    if echo "$TORCH_VER" | grep -q "${TORCH_VERSION}"; then
        echo "  PyTorch ${TORCH_VER} already installed"
    else
        retry_pip "pip install torch==${TORCH_VERSION}+${TORCH_CUDA} torchvision==0.22.0+${TORCH_CUDA} \
            --index-url https://download.pytorch.org/whl/${TORCH_CUDA}"
    fi
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"
    echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

#==============================================================================
# Step 5: Install Isaac Lab core + dependencies
#==============================================================================
log "Step 5/8: Isaac Lab Python packages"

# Critical: isaaclab core must be installed as editable package
# The --no-deps flag avoids build failures from flatdict's legacy setup.py
ISAACLAB_VER=$(python -c "from importlib.metadata import version; print(version('isaaclab'))" 2>/dev/null || echo "none")
if [ "$ISAACLAB_VER" != "none" ]; then
    echo "  isaaclab ${ISAACLAB_VER} already installed"
else
    pip install --no-deps -e "${ISAACLAB_PATH}/source/isaaclab"
fi

# Install sub-packages if missing
for pkg in isaaclab_assets isaaclab_tasks isaaclab_rl; do
    PKG_VER=$(python -c "from importlib.metadata import version; print(version('${pkg//_/-}'))" 2>/dev/null || echo "none")
    if [ "$PKG_VER" = "none" ]; then
        pip install --no-deps -e "${ISAACLAB_PATH}/source/${pkg}"
    fi
done

# rsl_rl (the RL training library)
RSL_VER=$(python -c "from importlib.metadata import version; print(version('rsl-rl-lib'))" 2>/dev/null || echo "none")
if [ "$RSL_VER" = "none" ]; then
    pip install rsl-rl-lib
fi

#==============================================================================
# Step 6: Install critical Python dependencies
#==============================================================================
log "Step 6/8: Python dependencies (pitfall fixes)"

# These are all packages that Isaac Lab / IsaacSim need but don't properly
# declare or that break due to version conflicts:

# flatdict: required by isaaclab.sim.simulation_context, not installed by --no-deps
pip install "flatdict>=4.0" 2>/dev/null | grep -v "already satisfied" || true

# prettytable: required by isaaclab.managers.action_manager
pip install "prettytable>=3.0" 2>/dev/null | grep -v "already satisfied" || true

# gymnasium: required for env registration and gym.make()
pip install "gymnasium>=1.0,<2.0" 2>/dev/null | grep -v "already satisfied" || true

# setuptools: must be < 80 to provide pkg_resources (omni.kit.pipapi needs it)
# setuptools >= 82 removed pkg_resources, breaking Isaac Sim extension loading
SETUPTOOLS_VER=$(python -c "from importlib.metadata import version; print(version('setuptools'))" 2>/dev/null)
if python -c "from packaging.version import Version; assert Version('${SETUPTOOLS_VER}') >= Version('80.0')" 2>/dev/null; then
    warn "setuptools ${SETUPTOOLS_VER} is too new (breaks pkg_resources). Downgrading..."
    pip install "setuptools>=69.0,<80.0"
fi

# Verify pkg_resources works
python -c "import pkg_resources" 2>/dev/null || {
    warn "pkg_resources missing, forcing setuptools reinstall..."
    pip install --force-reinstall "setuptools==69.5.1"
}

# onnxruntime: needed for Phase 3 ONNX export and benchmark
pip install onnxruntime 2>/dev/null | grep -v "already satisfied" || true

echo "  All critical deps OK"

#==============================================================================
# Step 7: Patch Isaac Lab kit config
#==============================================================================
log "Step 7/8: Patch kit config + system settings"

# Patch ALL kit files to remove unreliable cloudfront community registry
# (Isaac Sim hangs for minutes trying to connect, then sometimes crashes)
PATCHED=0
for kit_file in "${ISAACLAB_PATH}"/apps/*.kit; do
    if [ -f "$kit_file" ] && grep -q "dw290v42wisod.cloudfront.net" "$kit_file"; then
        sed -i '/dw290v42wisod.cloudfront.net/d' "$kit_file"
        echo "  Patched: $(basename $kit_file)"
        PATCHED=$((PATCHED + 1))
    fi
done
[ $PATCHED -eq 0 ] && echo "  Kit configs already patched"

# Disable Ubuntu "unresponsive application" dialog
# Isaac Sim blocks UI thread for 30-60s during extension loading, triggering this
if command -v gsettings >/dev/null 2>&1; then
    gsettings set org.gnome.mutter check-alive-timeout 0 2>/dev/null && \
        echo "  Disabled GNOME unresponsive dialog" || true
fi

# Pre-download G1 USD model (~22MB) to avoid GUI blocking on first launch
# Isaac Sim GUI mode synchronously downloads from S3 on first use, causing hangs
G1_USD_LOCAL="${EASYRL_ROOT}/applications/g1_locomotion/assets/g1_minimal.usd"
if [ ! -f "$G1_USD_LOCAL" ]; then
    log "  Pre-downloading G1 USD model..."
    G1_USD_URL="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd"
    mkdir -p "$(dirname $G1_USD_LOCAL)"
    if curl -fsSL --connect-timeout 30 "$G1_USD_URL" -o "$G1_USD_LOCAL"; then
        echo "  Downloaded G1 USD ($(du -h $G1_USD_LOCAL | cut -f1))"
    else
        warn "Failed to download G1 USD (non-fatal, will download on first GUI launch)"
        rm -f "$G1_USD_LOCAL"
    fi
else
    echo "  G1 USD already cached"
fi

#==============================================================================
# Step 8: Setup conda activate hooks
#==============================================================================
log "Step 8/8: Conda activate hooks"

ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/g1_locomotion.sh" << EOF
#!/bin/bash
# Auto-configured by g1_locomotion/setup_env.sh

export ISAACLAB_PATH="${ISAACLAB_PATH}"

# Add Isaac Lab source packages and easyRL to PYTHONPATH
export _G1_OLD_PYTHONPATH="\${PYTHONPATH:-}"
export PYTHONPATH="${ISAACLAB_PATH}/source/isaaclab:\
${ISAACLAB_PATH}/source/isaaclab_tasks:\
${ISAACLAB_PATH}/source/isaaclab_assets:\
${ISAACLAB_PATH}/source/isaaclab_rl:\
${EASYRL_ROOT}:\
\${PYTHONPATH:-}"
EOF

cat > "${DEACTIVATE_DIR}/g1_locomotion.sh" << EOF
#!/bin/bash
export PYTHONPATH="\${_G1_OLD_PYTHONPATH:-}"
unset _G1_OLD_PYTHONPATH
unset ISAACLAB_PATH
EOF

chmod +x "${ACTIVATE_DIR}/g1_locomotion.sh" "${DEACTIVATE_DIR}/g1_locomotion.sh"
echo "  Activate hook: ${ACTIVATE_DIR}/g1_locomotion.sh"

#==============================================================================
# Verification
#==============================================================================
log "Verification"

# Re-activate to pick up hooks
conda deactivate
conda activate "${CONDA_ENV_NAME}"

echo ""
echo "  Checking imports..."
python -c "
import sys, os
# Verify key packages
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'  PyTorch:     {torch.__version__} (CUDA: {torch.cuda.get_device_name(0)})')

from importlib.metadata import version
print(f'  IsaacSim:    {version(\"isaacsim\")}')
print(f'  isaaclab:    {version(\"isaaclab\")}')
print(f'  rsl_rl:      {version(\"rsl-rl-lib\")}')

# Verify Isaac Lab imports (before SimulationApp — these should work)
from isaaclab.app import AppLauncher
print(f'  AppLauncher: OK')

# Verify our custom modules
from applications.g1_locomotion.student.networks import AdaptationModule, StudentPolicy, StudentONNXWrapper
m = StudentONNXWrapper(AdaptationModule(100, 16), StudentPolicy(48, 16, 12))
out = m(torch.randn(1, 100), torch.randn(1, 48))
print(f'  G1 Networks: OK (output shape: {tuple(out.shape)})')

import flatdict, prettytable, pkg_resources, gymnasium, onnxruntime
print(f'  Dependencies: flatdict, prettytable, pkg_resources, gymnasium, onnxruntime OK')

# Verify SimulationApp is importable (not None)
import isaacsim
from isaacsim import SimulationApp
assert SimulationApp is not None, 'SimulationApp is None — exts/ missing'
print(f'  SimulationApp: {SimulationApp.__module__}')

print()
print('  All checks passed!')
"

#==============================================================================
# Summary
#==============================================================================
echo ""
echo "==========================================="
echo " G1 Locomotion Environment Ready"
echo "==========================================="
echo ""
echo " Quick start:"
echo "   conda activate ${CONDA_ENV_NAME}"
echo "   cd ${EASYRL_ROOT}"
echo ""
echo " Phase 1 — Train teacher:"
echo "   python applications/g1_locomotion/scripts/train_teacher.py \\"
echo "       --task G1-Flat-Custom-v0 --num_envs 1024 --headless"
echo ""
echo " Visualize:"
echo "   python applications/g1_locomotion/scripts/play.py \\"
echo "       --task G1-Flat-Custom-Play-v0 --load_run <run_dir>"
echo ""
echo " Phase 2 — Student distillation:"
echo "   python applications/g1_locomotion/scripts/collect_teacher_data.py \\"
echo "       --task G1-Flat-Custom-v0 --load_run <run_dir>"
echo "   python applications/g1_locomotion/student/train_student.py \\"
echo "       --data results/g1_flat_locomotion/teacher_distill_data.npz"
echo ""
echo " Phase 3 — ONNX export:"
echo "   python applications/g1_locomotion/export/export_onnx.py \\"
echo "       --student_path results/.../student/student_best.pt"
echo ""
echo " TensorBoard:"
echo "   tensorboard --logdir applications/g1_locomotion/results/"
echo ""
echo "==========================================="
