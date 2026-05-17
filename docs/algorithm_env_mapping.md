# Algorithm-Environment Mapping

| Algorithm | Environment | Action Space | Key Learning Focus |
|-----------|-------------|--------------|-------------------|
| Q-Learning | CliffWalking-v0 | Discrete | TD update, Q-table, exploration-exploitation |
| DQN | highway-env (discrete) | DiscreteMetaAction | Neural net replaces Q-table, experience replay, target network |
| Policy Gradient | highway-env (discrete) | DiscreteMetaAction | Policy gradient, weight log-prob by returns |
| PPO | highway-env (continuous) | ContinuousAction | Clip mechanism, Gaussian policy for continuous control |
| SAC | highway-env (continuous) | ContinuousAction | Maximum entropy framework, off-policy continuous control |

## Design Rationale

- **Q-Learning** uses CliffWalking as a minimal tabular environment to build intuition for TD learning.
- **DQN / PG** use highway-env with discrete actions (lane change, accelerate, decelerate) to demonstrate how neural networks handle continuous state spaces.
- **PPO / SAC** use highway-env with continuous actions (steering angle, throttle/brake) to demonstrate real vehicle control — the end goal for autonomous driving PnC engineers.

## Highway-env Configuration

- **Discrete version** (`make_lane_keeping_env`): `DiscreteMetaAction` — 5 meta-actions
- **Continuous version** (`make_continuous_lane_keeping_env`): `ContinuousAction` — steering + acceleration

Both versions share the same observation: Kinematics features (x, y, vx, vy, heading) for 5 surrounding vehicles.
