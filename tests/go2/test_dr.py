import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np


def test_dr_changes_physics():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer
    from applications.go2_locomotion.config import config

    env = Go2Env(config)
    env.reset()
    dr = Go2DomainRandomizer(env, config)

    original_mass = env.model.body_mass.copy()
    dr.randomize()
    assert not np.allclose(env.model.body_mass, original_mass, atol=1e-6)
    env.close()


def test_dr_returns_privileged_info():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer
    from applications.go2_locomotion.config import config

    env = Go2Env(config)
    env.reset()
    dr = Go2DomainRandomizer(env, config)
    priv = dr.randomize()
    assert priv.shape == (7,), f"Expected (7,) got {priv.shape}"
    # friction and mass should not be 0
    assert priv[0] != 0.0
    assert priv[1] != 0.0
    env.close()


def test_dr_step_applies_push():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer
    from applications.go2_locomotion.config import config

    # Use very short push interval to guarantee a push happens
    test_config = {**config, "dr_push_interval": [0.01, 0.02], "dr_ext_force_range": [5.0, 10.0]}
    env = Go2Env(test_config)
    env.reset()
    dr = Go2DomainRandomizer(env, test_config)
    dr.randomize()

    # Step many times to trigger push
    for _ in range(50):
        dr.step(test_config["control_dt"])

    # After pushes, ext_force should be nonzero
    assert np.any(dr._ext_force != 0.0)
    env.close()
