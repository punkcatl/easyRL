import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch


def test_actor_critic_output_shapes():
    from applications.go2_locomotion.agent.networks import AsymmetricActorCritic
    net = AsymmetricActorCritic(obs_dim=48, privileged_dim=7, action_dim=12, hidden_dim=128)
    obs = torch.randn(8, 48)
    priv = torch.randn(8, 7)
    actions, log_probs, values, entropy = net.act(obs, priv)
    assert actions.shape == (8, 12)
    assert log_probs.shape == (8,)
    assert values.shape == (8,)
    assert entropy.shape == ()


def test_ppo_update_runs_without_error():
    from applications.go2_locomotion.agent.ppo import PPOTrainer
    from applications.go2_locomotion.config import config

    test_config = {**config, "batch_size": 32}
    trainer = PPOTrainer(test_config)

    n = 128
    states = np.random.randn(n, 48).astype(np.float32)
    actions = np.random.randn(n, 12).astype(np.float32)
    rewards = np.random.randn(n).astype(np.float32)
    dones = np.zeros(n, dtype=bool)
    log_probs = np.random.randn(n).astype(np.float32)
    values = np.random.randn(n).astype(np.float32)
    privileged = np.zeros((n, 7), dtype=np.float32)

    trainer.update(states, actions, rewards, dones, log_probs, values,
                   next_value=0.0, privileged=privileged)


def test_ppo_act_returns_numpy():
    from applications.go2_locomotion.agent.ppo import PPOTrainer
    from applications.go2_locomotion.config import config

    trainer = PPOTrainer(config)
    obs = np.random.randn(4, 48).astype(np.float32)
    priv = np.zeros((4, 7), dtype=np.float32)
    actions, log_probs, values = trainer.act(obs, priv)
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (4, 12)
    assert log_probs.shape == (4,)
    assert values.shape == (4,)
