import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np


def test_vec_env_shapes():
    from applications.go2_locomotion.envs.vectorized import VecGo2Env
    from applications.go2_locomotion.config import config

    test_config = {**config, "num_envs": 4}
    vec_env = VecGo2Env(test_config)
    obs, infos = vec_env.reset()
    assert obs.shape == (4, 48)
    assert infos["privileged_obs"].shape == (4, 7)

    actions = np.zeros((4, 12), dtype=np.float32)
    obs, rewards, dones, truncs, infos = vec_env.step(actions)
    assert obs.shape == (4, 48)
    assert rewards.shape == (4,)
    assert dones.shape == (4,)
    vec_env.close()


def test_vec_env_auto_reset():
    from applications.go2_locomotion.envs.vectorized import VecGo2Env
    from applications.go2_locomotion.config import config

    test_config = {**config, "num_envs": 2, "episode_length_s": 0.1}
    vec_env = VecGo2Env(test_config)
    vec_env.reset()

    for _ in range(100):
        actions = np.random.uniform(-1, 1, (2, 12)).astype(np.float32)
        obs, rewards, dones, truncs, infos = vec_env.step(actions)
        assert not np.any(np.isnan(obs))
    vec_env.close()
