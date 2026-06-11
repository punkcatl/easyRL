#!/bin/bash
# One-click environment setup for unitree_rl_lab
# Clones all repos, installs all dependencies, configures everything.
#
# Tested on: Ubuntu 22.04, NVIDIA Driver 580+, RTX A4000
# Requirements: conda, git, NVIDIA driver
#
# Usage:
#   bash setup_env.sh                                    # repos go to easyRL's parent dir
#   bash setup_env.sh --workspace ~/projects             # repos go to ~/projects/
#   bash setup_env.sh --workspace ~/projects --env-name my_env
#
# Or download and run standalone:
#   curl -fsSL <raw-url> -o setup_env.sh && bash setup_env.sh --workspace ~/myRL

set -eo pipefail

#==============================================================================
# Configuration
#==============================================================================
CONDA_ENV_NAME="${CONDA_ENV_NAME:-env_isaaclab}"
PYTHON_VERSION="3.11"
ISAACLAB_VERSION="v2.3.0"
TORCH_VERSION="2.7.0"
TORCH_CUDA="cu128"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
WORKSPACE="${WORKSPACE:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd || pwd)}"
MAX_RETRIES=5
RETRY_DELAY=10

UNITREE_RL_LAB_REPO="https://github.com/unitreerobotics/unitree_rl_lab.git"
ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
UNITREE_ROS_REPO="https://github.com/unitreerobotics/unitree_ros.git"

#==============================================================================
# Parse arguments
#==============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --workspace) WORKSPACE="$(realpath "$2")"; shift 2 ;;
        --env-name) CONDA_ENV_NAME="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --workspace PATH   Root directory for all repos (default: cwd)"
            echo "  --env-name NAME    Conda env name (default: env_isaaclab)"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Directory layout after install:"
            echo "  <workspace>/"
            echo "    ├── unitree_rl_lab/   (this project)"
            echo "    ├── IsaacLab/         (Isaac Lab framework)"
            echo "    └── unitree_ros/      (robot URDF descriptions)"
            exit 0
            ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

UNITREE_RL_LAB_PATH="${WORKSPACE}/unitree_rl_lab"
ISAACLAB_PATH="${WORKSPACE}/IsaacLab"
UNITREE_ROS_PATH="${WORKSPACE}/unitree_ros"

#==============================================================================
# Helper functions
#==============================================================================
log() { echo -e "\n\033[1;32m[$(date '+%H:%M:%S')] $1\033[0m"; }
warn() { echo -e "\033[1;33m[WARN] $1\033[0m"; }
err() { echo -e "\033[1;31m[ERROR] $1\033[0m"; exit 1; }

retry_pip() {
    local cmd="$*"
    for i in $(seq 1 $MAX_RETRIES); do
        log "pip install attempt $i/$MAX_RETRIES"
        if eval "$cmd"; then
            return 0
        fi
        warn "Attempt $i failed. Retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    done
    err "pip install failed after $MAX_RETRIES attempts"
}

clone_if_missing() {
    local repo="$1" dest="$2" branch="${3:-}"
    if [ -d "$dest" ]; then
        echo "  Already exists: $dest"
    elif [ -n "$branch" ]; then
        git clone --branch "$branch" "$repo" "$dest"
    else
        git clone "$repo" "$dest"
    fi
}

#==============================================================================
# Pre-flight checks & auto-install prerequisites
#==============================================================================
log "Pre-flight checks"

command -v git >/dev/null 2>&1 || {
    warn "git not found, installing..."
    sudo apt-get update && sudo apt-get install -y git
}
command -v nvidia-smi >/dev/null 2>&1 || err "nvidia-smi not found. Install NVIDIA driver first."

# Auto-install Miniconda if conda is not available
if ! command -v conda >/dev/null 2>&1; then
    log "conda not found, installing Miniconda..."
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$MINICONDA_INSTALLER"
    bash "$MINICONDA_INSTALLER" -b -p "${HOME}/miniconda3"
    rm -f "$MINICONDA_INSTALLER"
    export PATH="${HOME}/miniconda3/bin:${PATH}"
    conda init bash >/dev/null 2>&1
    echo "Miniconda installed at ${HOME}/miniconda3"
fi

echo "GPU:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "Workspace: ${WORKSPACE}"

mkdir -p "${WORKSPACE}"

#==============================================================================
# Step 1: Clone all repositories
#==============================================================================
log "Step 1/8: Cloning repositories"

clone_if_missing "$UNITREE_RL_LAB_REPO" "$UNITREE_RL_LAB_PATH"
clone_if_missing "$ISAACLAB_REPO" "$ISAACLAB_PATH" "$ISAACLAB_VERSION"
clone_if_missing "$UNITREE_ROS_REPO" "$UNITREE_ROS_PATH"

#==============================================================================
# Step 2: Create conda environment
#==============================================================================
log "Step 2/8: Setting up conda environment '${CONDA_ENV_NAME}'"

eval "$(conda shell.bash hook)"

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    warn "Conda env '${CONDA_ENV_NAME}' already exists, reusing."
else
    conda create -n "${CONDA_ENV_NAME}" python=${PYTHON_VERSION} -y
fi
conda activate "${CONDA_ENV_NAME}"

#==============================================================================
# Step 3: Install IsaacSim
#==============================================================================
log "Step 3/8: Installing IsaacSim 5.1.0"

if python -c "import importlib.metadata; print(importlib.metadata.version('isaacsim'))" 2>/dev/null | grep -q "5.1"; then
    echo "IsaacSim already installed, skipping."
else
    pip install isaacsim==5.1.0.0 isaacsim-kernel==5.1.0.0
fi

#==============================================================================
# Step 4: Install PyTorch (with retry for unstable networks)
#==============================================================================
log "Step 4/8: Installing PyTorch ${TORCH_VERSION}+${TORCH_CUDA}"

if python -c "import torch; assert '${TORCH_VERSION}' in torch.__version__" 2>/dev/null; then
    echo "PyTorch ${TORCH_VERSION} already installed, skipping."
else
    retry_pip "pip install torch==${TORCH_VERSION}+${TORCH_CUDA} torchvision==0.22.0+${TORCH_CUDA} \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch; assert torch.cuda.is_available()" || err "CUDA not available in PyTorch!"

#==============================================================================
# Step 5: Install Isaac Lab
#==============================================================================
log "Step 5/8: Installing Isaac Lab"

export ISAACLAB_PATH
if python -c "import isaaclab_rl" 2>/dev/null; then
    echo "Isaac Lab already installed, skipping."
else
    cd "${ISAACLAB_PATH}"
    echo "Yes" | ./isaaclab.sh -i rsl_rl
fi

#==============================================================================
# Step 6: Install unitree_rl_lab
#==============================================================================
log "Step 6/8: Installing unitree_rl_lab"

cd "${UNITREE_RL_LAB_PATH}"
pip install -e source/unitree_rl_lab/

#==============================================================================
# Step 7: Configure robot model path
#==============================================================================
log "Step 7/8: Configuring UNITREE_ROS_DIR"

# Patch unitree.py if it still has the placeholder (for upstream compatibility)
UNITREE_PY="${UNITREE_RL_LAB_PATH}/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py"
if grep -q 'UNITREE_ROS_DIR = "path/to/unitree_ros"' "${UNITREE_PY}" 2>/dev/null; then
    sed -i "s|UNITREE_ROS_DIR = \"path/to/unitree_ros\".*|UNITREE_ROS_DIR = \"${UNITREE_ROS_PATH}\"|" "${UNITREE_PY}"
    echo "  Set UNITREE_ROS_DIR = ${UNITREE_ROS_PATH}"
fi

#==============================================================================
# Step 8: Setup conda activate hook
#==============================================================================
log "Step 8/8: Setting up conda activate hook"

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/unitree_rl_lab.sh" << EOF
#!/bin/bash
export ISAACLAB_PATH="${ISAACLAB_PATH}"
source "${UNITREE_RL_LAB_PATH}/unitree_rl_lab.sh"
EOF

#==============================================================================
# Final verification
#==============================================================================
log "Verifying installation"

echo "  PyTorch:        $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:           $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  IsaacSim:       $(python -c 'from importlib.metadata import version; print(version("isaacsim"))')"
echo "  isaaclab_rl:    $(python -c 'from importlib.metadata import version; print(version("isaaclab-rl"))')"
echo "  rsl_rl:         $(python -c 'from importlib.metadata import version; print(version("rsl-rl-lib"))')"
echo "  unitree_rl_lab: $(python -c 'from importlib.metadata import version; print(version("unitree_rl_lab"))')"

for robot in go2_description h1_description g1_description; do
    if [ -d "${UNITREE_ROS_PATH}/robots/${robot}" ]; then
        echo "  ${robot}: OK"
    else
        warn "${robot} not found!"
    fi
done

log "All done!"
echo ""
echo "========================================="
echo " Quick start:"
echo "========================================="
echo "  conda activate ${CONDA_ENV_NAME}"
echo "  cd ${UNITREE_RL_LAB_PATH}"
echo "  ./unitree_rl_lab.sh -l                              # list tasks"
echo "  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity  # train"
echo ""
echo "Note: First run of IsaacSim will prompt for EULA (type 'Yes')."
