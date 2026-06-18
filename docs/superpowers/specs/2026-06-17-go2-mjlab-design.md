# Go2 mjlab Migration Design

Migrate Go2 locomotion training from CPU MuJoCo to GPU-accelerated mjlab (MuJoCo Warp), targeting 20-50x training speedup on RTX A4000.

## Approach

Fork relevant code from `unitree_rl_mjlab` into `applications/go2_mjlab/`, using mjlab + RSL-RL as pip dependencies. Full control over task definition, reward, DR, and distillation pipeline.

## Hardware

- GPU: NVIDIA RTX A4000 (16GB VRAM, ~10GB available for training)
- Target: 1024-2048 parallel environments on GPU

## Directory Structure

```
applications/go2_mjlab/
├── README.md
├── pyproject.toml              # dependencies (mjlab, rsl-rl, mujoco-warp)
├── assets/
│   └── go2/                    # Go2 MJCF model (from unitree_rl_mjlab)
├── config/
│   ├── __init__.py
│   ├── go2_flat_cfg.py         # flat terrain env config
│   ├── go2_rough_cfg.py        # rough terrain env config
│   └── ppo_cfg.py              # RSL-RL PPO hyperparameters
├── envs/
│   ├── __init__.py
│   ├── go2_env.py              # mjlab manager-based env definition
│   ├── rewards.py              # reward functions (migrated from go2_locomotion)
│   ├── observations.py         # obs (48D) + privileged obs (7D)
│   └── domain_randomization.py # DR curriculum (3-phase)
├── train_teacher.py            # Phase 2: Teacher training (asymmetric AC + PPO)
├── distill_student.py          # Phase 3: offline distillation
├── export_onnx.py              # Phase 3: ONNX export
├── play.py                     # visualization / replay
└── scripts/
    ├── validate_env.py         # Phase 1: env sanity check
    └── benchmark_throughput.py # Phase 1: measure steps/sec on A4000
```

## Phase 1: Environment Validation

**Goal**: Confirm mjlab + MuJoCo Warp + Go2 runs on A4000.

### Tasks

1. Install mjlab, mujoco-warp, rsl-rl, warp-lang
2. Extract Go2 MJCF and scene config from unitree_rl_mjlab
3. Register Go2 task using mjlab manager-based API
4. Run 1024 envs, confirm no OOM
5. Benchmark steps/sec, compare to CPU baseline (~3K steps/sec)

### Success Criteria

- 1024 envs stable on A4000
- Throughput > 30K steps/sec (10x improvement over CPU)

## Phase 2: Teacher Training

**Goal**: Full Teacher pipeline with asymmetric actor-critic.

### Observation Space (48D)

Same as existing go2_locomotion:
- base_lin_vel (3)
- base_ang_vel (3)
- projected_gravity (3)
- joint_pos_relative (12)
- joint_vel (12)
- last_action (12)
- command (3)

### Privileged Observation (7D)

- friction (1)
- mass_scale (1)
- external_force (3)
- motor_strength (2)

### Asymmetric Actor-Critic

- Actor: obs(48D) → hidden(128) → action(12D)
- Critic: obs(48D) + privileged(7D) = 55D → hidden(128) → value(1)
- RSL-RL natively supports this via `num_privileged_obs` config

### Reward Design

Migrated directly from `go2_locomotion/config.py` Round 11:

```python
reward_scales = {
    "lin_vel_tracking": 3.0,
    "ang_vel_tracking": 0.3,
    "forward_progress": 1.5,
    "feet_air_time_reward": 1.0,
    "gait_schedule": 2.0,
    "gait_symmetry": 1.5,
    "all_feet_contact_penalty": -1.5,
    "base_height_reward": 0.5,
    "termination_penalty": -10.0,
    "flat_orientation_penalty": -0.5,
    "lin_vel_z_penalty": -2.0,
    "ang_vel_xy_penalty": -0.05,
    "action_rate_penalty": -0.1,
    "torque_penalty": -0.00005,
    "joint_acc_penalty": -2.5e-7,
    "collision_penalty": -1.0,
}
```

These will be implemented as mjlab reward manager terms.

### Domain Randomization Curriculum

Three phases (same as existing):
- Phase 1 (iter 0-500): No DR, learn basic locomotion
- Phase 2 (iter 500-1500): Light DR (friction [0.8,1.1], mass [0.95,1.05], force [0,1])
- Phase 3 (iter 1500+): Full DR (friction [0.5,1.25], mass [0.8,1.2], force [0,3], motor [0.9,1.1], PD gains [0.8,1.2])

Implemented via mjlab's randomization manager with phase switching based on iteration count.

### Command Curriculum

- Initial range: lin_vel_x [0.3, 0.8]
- Expand upper bound by 0.1 after 5 consecutive tracking_ratio > 0.6
- Limit: lin_vel_x max 1.5 m/s

### PPO Hyperparameters (for RSL-RL)

```python
lr = 3e-4
lr_end = 3e-5
gamma = 0.99
lam = 0.95
clip_param = 0.2
num_learning_epochs = 5
num_mini_batches = 4        # adapted for larger batch (1024 envs * 48 steps)
n_steps_per_env = 48
max_iterations = 5000
entropy_coef = 0.02
value_loss_coef = 1.0
max_grad_norm = 1.0
hidden_dims = [128, 128]
```

### Action Space

- 12D joint position targets (PD control)
- action_scale = 0.35
- PD gains: kp = [20,35,35]*4, kd = [1.0]*12
- Torque limit: 33.5 Nm

### Termination Conditions

- Body height < 0.20 or > 0.45
- Projected gravity z > -0.5 (fallen over)

## Phase 3: Student Distillation + ONNX Export

**Goal**: Offline teacher-student distillation, same as existing pipeline.

### Data Collection

- Load trained Teacher checkpoint
- Run Teacher policy in 1024 envs for ~500K transitions
- Store: `(obs_history[20], teacher_action)` pairs
- obs_history: rolling buffer of last 20 obs vectors (48D each) = input shape (20, 48)

### Student Architecture

- Input: obs_history (20, 48) = 960D flattened
- Encoder: MLP layers → latent (16D)
- Policy head: latent(16D) + current_obs(48D) → action(12D)
- Total input to policy: 64D

### Training

- Loss: MSE between student_action and teacher_action
- lr: 1e-3
- batch_size: 256
- epochs: 200 (with early stopping, patience=15)
- val_ratio: 0.1
- Dataset size: 500K transitions

### ONNX Export

- Export Student network (history encoder + policy head)
- opset_version: 17
- Input: obs_history (1, 20, 48)
- Output: action (1, 12)

## Dependencies

```
mjlab >= 1.4.0
mujoco >= 3.8.0
mujoco-warp >= 3.9.0
rsl-rl-lib >= 5.4.0
torch >= 2.7.0
warp-lang >= 1.14.0
```

## Expected Performance

| Metric | Current (CPU) | mjlab (A4000) |
|--------|--------------|---------------|
| num_envs | 128 | 1024-2048 |
| steps/sec | ~3K | ~60K-150K |
| Teacher 5000 iter | 3-6 hours | 10-20 min |
| Iteration turnaround | 2-3 rounds/day | 20+ rounds/day |

## Migration Notes

- Go2 MJCF from unitree_rl_mjlab should be compatible with MuJoCo Warp
- RSL-RL's OnPolicyRunner handles rollout collection, GAE, PPO update — replaces custom `PPOTrainer`
- mjlab's manager-based API replaces manual obs/reward/DR code with declarative config
- Joint ordering must be verified: ensure MJCF joint order matches existing FL/FR/RL/RR hip/thigh/calf convention
- PD control at 50Hz policy / 200Hz physics decimation preserved
