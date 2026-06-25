#!/bin/bash
# G1 mjlab environment setup script
# Creates conda environment with all dependencies for G1 humanoid training.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh

set -e

ENV_NAME="mjlab"
PYTHON_VERSION="3.11"

echo "=== G1 mjlab Environment Setup ==="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    echo "To reinstall: conda env remove -n ${ENV_NAME} && ./setup_env.sh"
    echo ""
    echo "Activating existing environment and installing g1_mjlab..."
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
    pip install -e . --no-deps
    echo ""
    echo "Done! Run: conda activate ${ENV_NAME}"
    exit 0
fi

echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo ""
echo "Installing PyTorch (CUDA 12.x)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "Installing MuJoCo + Warp..."
pip install mujoco>=3.8.0 mujoco-warp>=3.8.0 warp-lang>=1.14.0

echo ""
echo "Installing mjlab..."
pip install mjlab>=1.4.0

echo ""
echo "Installing RL + utilities..."
pip install rsl-rl-lib>=5.2.0 tyro>=1.0.1 tensorboard>=2.20.0

echo ""
echo "Installing ONNX export dependencies..."
pip install onnxscript>=0.5.4 onnxruntime

echo ""
echo "Installing g1_mjlab (editable)..."
pip install -e . --no-deps

echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import mujoco
print(f'MuJoCo: {mujoco.__version__}')
import mjlab
print(f'mjlab: {mjlab.__version__}')
import warp
print(f'Warp: {warp.__version__}')
print('All OK!')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo "  conda activate ${ENV_NAME}"
echo "  cd $(pwd)"
echo "  WANDB_MODE=disabled python scripts/train.py --num-envs 2048"
