import numpy as np
from envs.highway_lane_keeping import make_lane_keeping_env


def test_env_creates_successfully():
    env = make_lane_keeping_env()
    assert env is not None
    obs, info = env.reset()
    assert obs is not None
    env.close()


def test_env_observation_shape():
    env = make_lane_keeping_env()
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert len(obs.shape) >= 1
    env.close()


def test_env_step():
    env = make_lane_keeping_env()
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    assert next_obs is not None
    assert isinstance(reward, (int, float))
    env.close()
