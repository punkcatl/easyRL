# Go2 mjlab Pipeline Overview

## Project Flow

```
+-------------------------------------------------------------------+
|                      Phase 1: Teacher Training                     |
+-------------------------------------------------------------------+
|                                                                     |
|  Input: Go2 MJCF + env_cfg.py + rl_cfg.py + rewards.py            |
|                                                                     |
|  Script: scripts/train.py                                          |
|  Config: PPO + RSL-RL, 2048 envs, GPU (RTX A4000)                 |
|                                                                     |
|  Output: results_r9/model_3000.pt  [Teacher checkpoint]            |
|                                                                     |
|  Verify: play Go2-Flat-v0 --checkpoint-file <path> --num-envs 1   |
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
|  Script: src/distill/collect_data.py                               |
|  Config: 500K transitions, history_length=20                       |
|                                                                     |
|  Output: results/distill_dataset.npz  [Distillation dataset]       |
|          obs_history: [500K, 20, 50]  (50D = 48D obs + 2D phase)   |
|          actions:     [500K, 12]                                    |
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
|  Script: src/distill/train_student.py                              |
|  Config: MLP encoder (960D->128->16) + policy head (64D->128->12) |
|          MSE loss, lr=1e-3, early stopping (patience=15)           |
|                                                                     |
|  Output: results/student_final.pt  [Student checkpoint]            |
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
|  Script: src/distill/export_onnx.py                                |
|  Config: opset_version=17, dynamic batch axis                      |
|                                                                     |
|  Output: results/student_policy.onnx  [Final deployment model]     |
|          Input:  obs_history [batch, 20, 50]                        |
|          Output: action [batch, 12]                                 |
|                                                                     |
+-------------------------------------------------------------------+
```

## Artifacts

| Phase | Artifact | Path | Type |
|-------|----------|------|------|
| 1 | Teacher model | `results_r9/model_*.pt` | Trained model |
| 2 | Distillation dataset | `results/distill_dataset.npz` | Data |
| 3 | Student model | `results/student_final.pt` | Trained model |
| 4 | ONNX deployment model | `results/student_policy.onnx` | Trained model (final) |

## Current Status

- Phase 1: Complete (R9, 3000 iter, reward 84.77, vel_tracking 97.9%)
- Phase 2-4: Code ready, pending teacher quality improvements (gait shaping, DR)

## Commands

Prerequisites:
```bash
conda activate mjlab
cd ~/Desktop/myRL/easyRL/applications/go2_mjlab
export WANDB_MODE=disabled
```

Pipeline:
```bash
# Phase 1: Train teacher
python scripts/train.py --num-envs 2048 --max-iterations 3000 --log-dir results_r9

# Visualize trained policy
play Go2-Flat-v0 --checkpoint-file results_r9/model_3000.pt --num-envs 1

# Phase 2: Collect distillation data
python -m src.distill.collect_data \
    --checkpoint results_r9/model_3000.pt \
    --num-transitions 500000

# Phase 3: Train student
python -m src.distill.train_student \
    --dataset-path results/distill_dataset.npz

# Phase 4: Export ONNX
python -m src.distill.export_onnx \
    --checkpoint results/student_final.pt
```
