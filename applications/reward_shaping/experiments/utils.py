"""Shared training loop for all reward shaping experiments."""
import numpy as np


def train_ppo(env, agent, n_episodes, verbose_interval=100, label=""):
    """Run PPO training loop for n_episodes, returning per-episode returns.

    Handles both continuous and discrete action spaces, and properly
    bootstraps value for truncated (but not terminated) episodes.

    Args:
        env: Gymnasium environment (already wrapped with reward shaping).
        agent: PPOAgent instance.
        n_episodes: Number of episodes to train.
        verbose_interval: Print average return every N episodes (0 = silent).
        label: Label string for verbose printing.

    Returns:
        List of per-episode undiscounted returns.
    """
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        states, actions, rewards_buf, log_probs, values = [], [], [], [], []
        terminations, dones = [], []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, log_prob, value = agent.take_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards_buf.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            terminations.append(terminated)          # True only on natural termination
            dones.append(terminated or truncated)    # True on either (for loop control)
            state = next_obs.flatten()

        # Bootstrap value for truncated (time-limit) episodes
        if truncated and not terminated:
            next_value = agent.take_action(state)[2]
        else:
            next_value = 0.0

        agent.update(states, actions, rewards_buf, log_probs, values, terminations, next_value)
        returns.append(sum(rewards_buf))

        if verbose_interval and (ep + 1) % verbose_interval == 0:
            avg = np.mean(returns[-verbose_interval:])
            print(f"    {label}Episode {ep+1}/{n_episodes} | Avg Return: {avg:.2f}")

    return returns
