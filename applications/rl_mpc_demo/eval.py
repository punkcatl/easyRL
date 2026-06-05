import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from config import config
from envs.highway_wrapper import HighwayEnvWrapper
from controller.lon_mpc import LonMPC
from controller.lat_mpc import LatMPC
from controller.action_mapper import ActionMapper, ACTION_NAMES
from agent.ppo_decision import PPODecisionAgent


def evaluate(env_id: str = None, model_path: str = None,
             n_episodes: int = 5, render: bool = True):
    env_id = env_id or config["env_id"]
    results_dir = Path(__file__).resolve().parent / "results"

    if model_path is None:
        model_path = str(results_dir / f"ppo_mpc_{env_id.replace('-', '_')}.pth")

    render_mode = "human" if render else None
    env = HighwayEnvWrapper(env_id, render_mode=render_mode)

    agent = PPODecisionAgent(
        state_dim=env.observation_dim, action_dim=5,
        hidden_dim=config["ppo_hidden_dim"],
    )
    agent.load(model_path)

    lon_mpc = LonMPC(
        N=config["lon_N"], dt=config["lon_dt"],
        Q_v=config["lon_Q_v"], Q_a=config["lon_Q_a"], R_j=config["lon_R_j"],
        a_min=config["a_min"], a_max=config["a_max"],
        j_min=config["j_min"], j_max=config["j_max"],
        v_min=config["v_min"], v_max=config["v_max"],
    )

    lat_mpc = LatMPC(
        N=config["lat_N"], dt=config["lat_dt"], L=config["wheelbase"],
        Q_y=config["lat_Q_y"], Q_psi=config["lat_Q_psi"],
        R_delta=config["lat_R_delta"],
        delta_min=config["delta_min"], delta_max=config["delta_max"],
        delta_dot_max=config["delta_dot_max"],
    )

    mapper = ActionMapper(
        delta_v=config["delta_v"], v_min=config["v_min"],
        v_max=config["v_max"], lane_width=config["lane_width"],
    )

    all_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        ego = env.get_ego_state()
        mapper.reset(ego["speed"], ego["y"], lane_center_fn=env.get_lane_center_y)
        lat_mpc.reset()
        lon_mpc.reset()

        episode_reward = 0
        done = False
        a_current = 0.0

        # Logging for plots
        log_v, log_v_ref = [], []
        log_y, log_y_ref = [], []
        log_actions = []

        while not done:
            action = agent.take_action(obs)
            v_ref, y_ref = mapper.map(
                action, current_y=ego["y"],
                lane_center_fn=env.get_lane_center_y,
            )

            a_des = lon_mpc.solve(
                s=ego["x"], v=ego["speed"], a=a_current, v_ref=v_ref
            )
            delta = lat_mpc.solve(
                x=ego["x"], y=ego["y"], psi=ego["heading"],
                v=ego["speed"], y_ref=y_ref,
                psi_ref=env.get_road_heading(),
            )

            steering_normalized = np.clip(delta / config["delta_max"], -1.0, 1.0)
            if a_des >= 0:
                accel_normalized = np.clip(a_des / config["a_max"], 0.0, 1.0)
            else:
                accel_normalized = np.clip(a_des / abs(config["a_min"]), -1.0, 0.0)

            next_obs, reward, done, info, _, _ = env.step(
                steering_normalized, accel_normalized
            )

            # Log
            log_v.append(ego["speed"])
            log_v_ref.append(v_ref)
            log_y.append(ego["y"])
            log_y_ref.append(y_ref)
            log_actions.append(action)

            obs = next_obs
            ego = env.get_ego_state()
            a_current = a_des
            episode_reward += reward

        all_rewards.append(episode_reward)
        print(f"Episode {ep+1}: reward = {episode_reward:.2f}")

        # Plot tracking performance for last episode
        if ep == n_episodes - 1:
            _plot_tracking(log_v, log_v_ref, log_y, log_y_ref,
                           log_actions, env_id, results_dir)

    env.close()
    print(f"\nAverage reward over {n_episodes} episodes: {np.mean(all_rewards):.2f}")


def _plot_tracking(log_v, log_v_ref, log_y, log_y_ref,
                   log_actions, env_id, results_dir):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = np.arange(len(log_v))

    axes[0].plot(t, log_v, label="v_actual")
    axes[0].plot(t, log_v_ref, "--", label="v_ref")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend()
    axes[0].set_title(f"RL+MPC Tracking Performance ({env_id})")

    axes[1].plot(t, log_y, label="y_actual")
    axes[1].plot(t, log_y_ref, "--", label="y_ref")
    axes[1].set_ylabel("Lateral Position (m)")
    axes[1].legend()

    axes[2].step(t, log_actions, where="post")
    axes[2].set_ylabel("PPO Action")
    axes[2].set_xlabel("Step")
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(ACTION_NAMES)

    plt.tight_layout()
    save_path = results_dir / f"tracking_{env_id.replace('-', '_')}.png"
    plt.savefig(str(save_path), dpi=150)
    print(f"Tracking plot saved to {save_path}")
    plt.close()


def plot_training_curve(env_id: str = None):
    env_id = env_id or config["env_id"]
    results_dir = Path(__file__).resolve().parent / "results"
    returns = np.load(str(results_dir / f"returns_{env_id.replace('-', '_')}.npy"))

    plt.figure(figsize=(10, 5))
    plt.plot(returns, alpha=0.3, label="Episode Return")
    # Moving average
    window = 50
    if len(returns) >= window:
        ma = np.convolve(returns, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(returns)), ma, label=f"Moving Avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"PPO+MPC Training Curve ({env_id})")
    plt.legend()
    plt.tight_layout()
    save_path = results_dir / f"training_curve_{env_id.replace('-', '_')}.png"
    plt.savefig(str(save_path), dpi=150)
    print(f"Training curve saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate PPO+MPC on highway-env")
    parser.add_argument("--env", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--plot-training", action="store_true")
    args = parser.parse_args()

    if args.plot_training:
        plot_training_curve(args.env)
    else:
        evaluate(env_id=args.env, model_path=args.model,
                 n_episodes=args.episodes, render=not args.no_render)
