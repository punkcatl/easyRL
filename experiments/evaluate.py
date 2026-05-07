import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import numpy as np
from utils.metrics import compute_control_metrics


def evaluate_agent(env, agent, n_episodes: int = 20, flatten_obs: bool = True) -> dict:
    """Evaluate a trained agent and compute control quality metrics."""
    all_rewards = []
    all_lateral = []
    all_heading = []
    all_steering = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten() if flatten_obs else obs
        episode_reward = 0
        episode_lateral = []
        episode_heading = []
        episode_steering = []
        done = False
        prev_steering = 0.0

        while not done:
            # Get action from agent
            action = agent.select_action(state)
            # Handle tuple returns (PPO returns action, log_prob, value)
            if isinstance(action, tuple):
                action = action[0]

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Extract control metrics from environment
            try:
                ego = env.unwrapped.vehicle
                if ego is not None:
                    # Lateral deviation from lane center
                    lane = ego.lane
                    if lane is not None:
                        lane_coords = lane.local_coordinates(ego.position)
                        episode_lateral.append(lane_coords[1])
                    # Heading
                    episode_heading.append(ego.heading)
                    # Steering (approximate from action)
                    episode_steering.append(float(action) if np.isscalar(action) else float(action[0]) if hasattr(action, '__len__') else 0.0)
            except Exception:
                pass

            state = next_obs.flatten() if flatten_obs else next_obs
            episode_reward += reward

        all_rewards.append(episode_reward)
        all_lateral.extend(episode_lateral)
        all_heading.extend(episode_heading)
        all_steering.extend(episode_steering)

    # Compute metrics
    if all_lateral and all_heading and all_steering:
        control_metrics = compute_control_metrics(all_lateral, all_heading, all_steering)
    else:
        control_metrics = {
            "lateral_mean": 0.0, "lateral_std": 0.0,
            "heading_mean": 0.0, "steering_smoothness": 0.0,
        }
    control_metrics["mean_reward"] = float(np.mean(all_rewards))
    control_metrics["std_reward"] = float(np.std(all_rewards))
    return control_metrics
