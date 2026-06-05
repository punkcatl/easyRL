import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from tqdm import tqdm

from config import config
from envs.highway_wrapper import HighwayEnvWrapper
from controller.lon_mpc import LonMPC
from controller.lat_mpc import LatMPC
from controller.action_mapper import ActionMapper
from agent.ppo_decision import PPODecisionAgent


def train(env_id: str = None, n_episodes: int = None, render: bool = False):
    env_id = env_id or config["env_id"]
    n_episodes = n_episodes or config["n_episodes"]

    # Environment
    render_mode = "human" if render else None
    env = HighwayEnvWrapper(env_id, render_mode=render_mode)

    # PPO agent
    agent = PPODecisionAgent(
        state_dim=env.observation_dim,
        action_dim=5,
        hidden_dim=config["ppo_hidden_dim"],
        lr=config["ppo_lr"],
        gamma=config["ppo_gamma"],
        lmbda=config["ppo_lmbda"],
        eps=config["ppo_eps"],
        epochs=config["ppo_epochs"],
    )

    # MPC controllers
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

    # Action mapper
    mapper = ActionMapper(
        delta_v=config["delta_v"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        lane_width=config["lane_width"],
    )

    # Training
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    return_list = []

    for episode in tqdm(range(n_episodes), desc=f"Training PPO+MPC on {env_id}"):
        obs = env.reset()
        ego = env.get_ego_state()
        mapper.reset(ego["speed"], ego["y"], lane_center_fn=env.get_lane_center_y)
        lat_mpc.reset()
        lon_mpc.reset()

        transition_dict = {
            'states': [], 'actions': [], 'next_states': [],
            'rewards': [], 'dones': [], 'terminations': []
        }
        episode_reward = 0
        done = False
        a_current = 0.0

        while not done:
            # PPO decision
            action = agent.take_action(obs)

            # Map to references
            v_ref, y_ref = mapper.map(
                action,
                current_y=ego["y"],
                lane_center_fn=env.get_lane_center_y,
            )

            # MPC control
            a_des = lon_mpc.solve(
                s=ego["x"], v=ego["speed"], a=a_current, v_ref=v_ref
            )
            delta = lat_mpc.solve(
                x=ego["x"], y=ego["y"], psi=ego["heading"],
                v=ego["speed"], y_ref=y_ref,
                psi_ref=env.get_road_heading(),
            )

            # Normalize to highway-env action range [-1, 1]
            steering_normalized = np.clip(delta / config["delta_max"], -1.0, 1.0)
            if a_des >= 0:
                accel_normalized = np.clip(a_des / config["a_max"], 0.0, 1.0)
            else:
                accel_normalized = np.clip(a_des / abs(config["a_min"]), -1.0, 0.0)

            # Step environment
            next_obs, reward, done, info, terminated, truncated = env.step(
                steering_normalized, accel_normalized
            )

            # Store transition
            transition_dict['states'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_obs)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            transition_dict['terminations'].append(terminated)

            obs = next_obs
            ego = env.get_ego_state()
            a_current = a_des
            episode_reward += reward

        # Update PPO
        agent.update(transition_dict)
        return_list.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg = np.mean(return_list[-50:])
            print(f"  Episode {episode+1}/{n_episodes} | Avg Reward (50): {avg:.2f}")

    # Save
    agent.save(str(results_dir / f"ppo_mpc_{env_id.replace('-', '_')}.pth"))
    np.save(str(results_dir / f"returns_{env_id.replace('-', '_')}.npy"), return_list)

    env.close()
    print(f"Training complete. Results saved to {results_dir}/")
    return return_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO+MPC on highway-env")
    parser.add_argument("--env", default=None, help="Environment ID")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(env_id=args.env, n_episodes=args.episodes, render=args.render)
