import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest


def test_env_creation():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config
    env = Go2Env(config)
    assert env.observation_space.shape == (48,)
    assert env.action_space.shape == (12,)
    env.close()


def test_reset_returns_valid_obs():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config
    env = Go2Env(config)
    obs, info = env.reset()
    assert obs.shape == (48,)
    assert not np.any(np.isnan(obs))
    assert "privileged_obs" in info
    assert info["privileged_obs"].shape == (7,)
    env.close()


def test_step_returns_valid():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config
    env = Go2Env(config)
    env.reset()
    action = np.zeros(12, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (48,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert "reward_components" in info
    env.close()


def test_pd_control_moves_joints():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config
    env = Go2Env(config)
    env.reset()
    pos_before = env._get_joint_positions().copy()
    action = np.ones(12, dtype=np.float32) * 0.5
    for _ in range(10):
        env.step(action)
    pos_after = env._get_joint_positions().copy()
    assert np.any(np.abs(pos_after - pos_before) > 0.01)
    env.close()
