import warnings
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import highway_env  # noqa: F401 — registers highway environments

warnings.filterwarnings("ignore", message=".*unconventional shape.*")


def make_lane_keeping_env(render_mode: str = None) -> gym.Env:
    """Create a highway-v0 environment configured for lane keeping with discrete actions."""
    env = gym.make("highway-v0", render_mode=render_mode)
    env.configure(
        {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy", "heading"],
                "vehicles_count": 5,
                "absolute": False,
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 10,
            "duration": 60,
            "policy_frequency": 5,
            "screen_width": 1600,
            "screen_height": 500,
            "scaling": 15,
        }
    )
    # Close stale viewer so it gets recreated with new config on next reset
    if hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        env.unwrapped.viewer.close()
        env.unwrapped.viewer = None
    env = FlattenObservation(env)
    env.reset()
    return env


def make_continuous_lane_keeping_env(render_mode: str = None) -> gym.Env:
    """Create a highway-v0 environment configured for lane keeping with continuous actions."""
    env = gym.make("highway-v0", render_mode=render_mode)
    env.configure(
        {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy", "heading"],
                "vehicles_count": 5,
                "absolute": False,
            },
            "action": {
                "type": "ContinuousAction",
            },
            "lanes_count": 3,
            "vehicles_count": 10,
            "duration": 60,
            "policy_frequency": 5,
            "screen_width": 1600,
            "screen_height": 500,
            "scaling": 15,
        }
    )
    # Close stale viewer so it gets recreated with new config on next reset
    if hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        env.unwrapped.viewer.close()
        env.unwrapped.viewer = None
    env = FlattenObservation(env)
    env.reset()
    return env
