#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────
#  easyRL — one-shot environment installer
#  Usage:
#    bash install.sh          # auto-detect network, use mirror if slow
#    bash install.sh --mirror # force domestic mirror (pip + conda)
#    bash install.sh --no-mirror # force official source
# ─────────────────────────────────────────────

MIRROR_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --mirror)    MIRROR_FLAG="yes" ;;
    --no-mirror) MIRROR_FLAG="no"  ;;
  esac
done

# ── color helpers ─────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[info]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── network probe ─────────────────────────────
need_mirror() {
  if [[ "$MIRROR_FLAG" == "yes" ]]; then return 0; fi
  if [[ "$MIRROR_FLAG" == "no"  ]]; then return 1; fi
  info "Probing network speed to pypi.org ..."
  local t
  t=$(curl -o /dev/null -s -w "%{time_total}" --max-time 6 https://pypi.org/simple/pip/ 2>/dev/null || echo "999")
  # bc may not be installed; use awk for float comparison
  if awk "BEGIN{exit !($t > 3.0)}"; then
    warn "pypi.org response time ${t}s > 3s — switching to domestic mirrors"
    return 0
  fi
  info "Network OK (${t}s), using official sources"
  return 1
}

USE_MIRROR=false
need_mirror && USE_MIRROR=true

# ── mirror URLs ───────────────────────────────
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
CONDA_CHANNELS="-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
                -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free \
                -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch \
                -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"

pip_install() {
  if $USE_MIRROR; then
    pip install -i "$PIP_MIRROR" "$@"
  else
    pip install "$@"
  fi
}

# ── locate project root (script may be called from anywhere) ─
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ════════════════════════════════════════════
#  PATH A — conda available  →  environment.yml
# ════════════════════════════════════════════
if command -v conda &>/dev/null; then
  info "conda detected — using environment.yml (includes CUDA-aware PyTorch)"

  ENV_NAME="easyrl"

  # configure tsinghua mirror for conda if needed
  if $USE_MIRROR; then
    info "Configuring conda to use Tsinghua mirror"
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    conda config --set show_channel_urls yes
  fi

  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    warn "Conda env '$ENV_NAME' already exists — updating"
    conda env update -n "$ENV_NAME" -f environment.yml --prune
  else
    conda env create -f environment.yml
  fi

  # activate and finish with pip extras
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"

  info "Installing pip-only extras (onnx, netron) ..."
  pip_install onnx>=1.14 netron>=7.0

  info "Installing easyRL package in editable mode ..."
  pip_install -e .

# ════════════════════════════════════════════
#  PATH B — pip only  →  requirements.txt
# ════════════════════════════════════════════
else
  warn "conda not found — falling back to pip + requirements.txt"
  warn "For CUDA support, install conda first: https://docs.conda.io/en/latest/miniconda.html"

  # require python 3.9+
  python_ver=$(python3 -c "import sys; print(sys.version_info >= (3,9))")
  [[ "$python_ver" == "True" ]] || error "Python >= 3.9 required. Current: $(python3 --version)"

  info "Installing dependencies from requirements.txt ..."
  pip_install -r requirements.txt

  info "Installing easyRL package in editable mode ..."
  pip_install -e .
fi

# ════════════════════════════════════════════
#  Verify
# ════════════════════════════════════════════
info "Running smoke test ..."
python3 - <<'PYEOF'
import sys
import torch
import gymnasium
import highway_env  # noqa: F401
import pygame
import tqdm

print(f"  Python      : {sys.version.split()[0]}")
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
print(f"  Gymnasium   : {gymnasium.__version__}")
print(f"  pygame      : {pygame.__version__}")
print(f"  tqdm        : {tqdm.__version__}")
print("  highway-env : OK")
PYEOF

echo ""
info "Installation complete."
if command -v conda &>/dev/null; then
  echo -e "  Activate env : ${GREEN}conda activate easyrl${NC}"
fi
echo -e "  Quick start  : ${GREEN}python algorithms/dqn/train.py${NC}"
