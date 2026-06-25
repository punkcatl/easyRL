# G1 mjlab Pipeline Overview

## Project Flow

```
+-------------------------------------------------------------------+
|                      Phase 1: Teacher Training                     |
+-------------------------------------------------------------------+
|                                                                     |
|  Input: G1 MJCF (mjlab asset_zoo) + env_cfg.py + rl_cfg.py        |
|  Robot: 15 DOF (12 legs + 3 waist), arms PD-held                  |
|                                                                     |
|  Script: scripts/train.py                                          |
|  Config: PPO + RSL-RL, 2048 envs, GPU (RTX A4000)                 |
|                                                                     |
|  Output: results/model_5000.pt  [Teacher checkpoint]               |
|                                                                     |
|  Verify: python scripts/play.py --checkpoint results/model_5000.pt |
|                                                                     |
+-------------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------------+
|                      Phase 2: Data Collection                      |
+-------------------------------------------------------------------+
|                                                                     |
|  Input: Teacher checkpoint + environment (1024 envs)               |
|                                                                     |
|  Script: python -m src.distill.collect_data                        |
|  Config: 500K transitions, history_length=20                       |
|                                                                     |
|  Output: results/distill_dataset.npz                               |
|          obs_history: [500K, 20, 87]                               |
|          actions:     [500K, 15]                                    |
|                                                                     |
+-------------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------------+
|                    Phase 3: Student Distillation                    |
+-------------------------------------------------------------------+
|                                                                     |
|  Input: distill_dataset.npz                                        |
|                                                                     |
|  Script: python -m src.distill.train_student                       |
|  Config: MLP encoder (20*87 -> 256 -> 32) +                       |
|          policy head (32+87 -> 128 -> 15)                          |
|          MSE loss, lr=1e-3, early stopping (patience=15)           |
|                                                                     |
|  Output: results/student_final.pt                                  |
|                                                                     |
+-------------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------------+
|                       Phase 4: ONNX Export                          |
+-------------------------------------------------------------------+
|                                                                     |
|  Input: student_final.pt                                           |
|                                                                     |
|  Script: python -m src.distill.export_onnx                         |
|  Config: opset_version=17, dynamic batch axis                      |
|                                                                     |
|  Output: results/student_policy.onnx                               |
|          Input:  obs_history [batch, 20, 87]                       |
|          Output: action [batch, 15]                                 |
|                                                                     |
+-------------------------------------------------------------------+
```

## Commands

Prerequisites:
```bash
conda activate mjlab
cd ~/Desktop/myRL/easyRL/applications/g1_mjlab
export WANDB_MODE=disabled
```

Pipeline:
```bash
# Phase 1: Train teacher
python scripts/train.py --num-envs 2048 --max-iterations 5000 --log-dir results

# Visualize trained policy
python scripts/play.py --checkpoint results/model_5000.pt

# Phase 2: Collect distillation data
python -m src.distill.collect_data \
    --checkpoint results/model_5000.pt \
    --num-transitions 500000

# Phase 3: Train student
python -m src.distill.train_student \
    --dataset-path results/distill_dataset.npz

# Phase 4: Export ONNX
python -m src.distill.export_onnx \
    --checkpoint results/student_final.pt
```

## Key Differences from Go2 mjlab

| Dimension | Go2 mjlab | G1 mjlab |
|-----------|-----------|----------|
| Robot | Quadruped (12 DOF) | Humanoid (15 DOF) |
| Gait | Diagonal trot | Bipedal alternating |
| Base height | 0.34m | 0.74m |
| Gait period | 0.5s | 0.6s |
| Network | 512-256-128 | 512-256-128 |
| Student latent | 16D | 32D |
| Contact sensor | 4 feet | 2 feet (L/R groups) |
| Termination angle | 60 deg | 45 deg |
