"""G1 locomotion environment and agent configurations."""

import gymnasium as gym

from .flat_env_cfg import G1FlatLocomotionEnvCfg, G1FlatLocomotionEnvCfg_PLAY
from .rough_env_cfg import G1RoughLocomotionEnvCfg, G1RoughLocomotionEnvCfg_PLAY
from .ppo_cfg import G1FlatPPOCfg, G1RoughPPOCfg

gym.register(
    id="G1-Flat-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FlatLocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": G1FlatPPOCfg,
    },
)

gym.register(
    id="G1-Flat-Custom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FlatLocomotionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": G1FlatPPOCfg,
    },
)

gym.register(
    id="G1-Rough-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1RoughLocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": G1RoughPPOCfg,
    },
)

gym.register(
    id="G1-Rough-Custom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1RoughLocomotionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": G1RoughPPOCfg,
    },
)
