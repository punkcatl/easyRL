#!/bin/bash
# Go2 Locomotion Project Setup Script
# Run from the easyRL root directory:
#   bash applications/go2_locomotion/setup.sh

set -e

EASYRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS_DIR="$EASYRL_ROOT/applications/go2_locomotion/assets"

echo "========================================"
echo "  Go2 Locomotion Setup"
echo "  Root: $EASYRL_ROOT"
echo "========================================"

# ── Step 1: Python dependencies ──────────────────────────────────────────────
echo ""
echo "[1/3] Installing Python dependencies..."

pip install mujoco>=2.3.7
pip install gymnasium>=0.28.0
pip install torch>=2.0.0
pip install numpy
pip install onnx onnxruntime
pip install imageio    # required for mujoco rendering; mujoco.viewer also needs this

echo "      Dependencies installed."

# ── Step 2: Official Go2 MJCF model ──────────────────────────────────────────
echo ""
echo "[2/3] Fetching official Unitree Go2 MJCF model..."
echo "      (using git sparse-checkout to avoid downloading full menagerie)"

MENAGERIE_TMP="$(mktemp -d)/mujoco_menagerie"

if git clone \
    --depth=1 \
    --filter=blob:none \
    --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git \
    "$MENAGERIE_TMP" 2>&1; then

    cd "$MENAGERIE_TMP"
    git sparse-checkout set unitree_go2
    cd "$EASYRL_ROOT"

    # Verify the model loads correctly before copying
    python -c "
import mujoco
model = mujoco.MjModel.from_xml_path('$MENAGERIE_TMP/unitree_go2/scene.xml')
assert model.nq == 19 and model.nv == 18 and model.nu == 12, \
    f'Unexpected model dims: nq={model.nq} nv={model.nv} nu={model.nu}'
print('      Go2 model verified: nq=19, nv=18, nu=12')
"
    # Copy to project assets (overwrite approximate model with official one)
    cp -r "$MENAGERIE_TMP/unitree_go2/"* "$ASSETS_DIR/"
    echo "      Official Go2 model installed to assets/"

    rm -rf "$MENAGERIE_TMP"

else
    echo "      WARNING: Could not reach GitHub (air-gapped network?)."
    echo "      Falling back to approximate Go2 model (go2_scene.xml)."
    echo "      The approximate model has correct kinematics but simplified geometry."
    echo "      To use the official model later, run manually:"
    echo "        git clone --depth=1 --filter=blob:none --sparse \\"
    echo "          https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie"
    echo "        cd /tmp/menagerie && git sparse-checkout set unitree_go2"
    echo "        cp -r /tmp/menagerie/unitree_go2/* $ASSETS_DIR/"

    # Ensure go2_env.py points to fallback model
    sed -i 's|"assets" / "scene.xml"|"assets" / "go2_scene.xml"|g' \
        "$EASYRL_ROOT/applications/go2_locomotion/envs/go2_env.py" 2>/dev/null || true
fi

# ── Step 3: Smoke test ────────────────────────────────────────────────────────
echo ""
echo "[3/3] Running smoke test..."

cd "$EASYRL_ROOT"
python -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import mujoco
import mujoco.viewer   # must be imported explicitly on mujoco 2.3.x

from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.config import config
from applications.go2_locomotion.agent.ppo import PPOTrainer

# Env
env = Go2Env(config)
obs, info = env.reset()
assert obs.shape == (48,), f'obs shape wrong: {obs.shape}'
action = np.zeros(12, dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
print('      Go2Env: OK')

# PPO
trainer = PPOTrainer(config)
obs_batch = np.random.randn(4, 48).astype('float32')
priv_batch = np.zeros((4, 7), dtype='float32')
actions, lp, vals = trainer.act(obs_batch, priv_batch)
assert actions.shape == (4, 12)
print('      PPOTrainer: OK')

print('      Smoke test PASSED')
"

echo ""
echo "========================================"
echo "  Setup complete! Quick start:"
echo ""
echo "  # Train teacher (Phase 1, ~1.5h)"
echo "  python applications/go2_locomotion/train_teacher.py"
echo ""
echo "  # Evaluate with rendering"
echo "  python applications/go2_locomotion/evaluate.py --mode teacher --render"
echo ""
echo "  # Run Motion Test Suite"
echo "  python applications/go2_locomotion/benchmark.py --tag v1"
echo "========================================"
