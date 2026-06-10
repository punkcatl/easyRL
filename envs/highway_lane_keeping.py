import warnings
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import highway_env  # noqa: F401 — registers highway environments

warnings.filterwarnings("ignore", message=".*unconventional shape.*")


def make_racetrack_env(render_mode: str = None) -> gym.Env:
    """Create a racetrack-v0 environment with continuous lateral control for PPO training."""
    env = gym.make("racetrack-v0", render_mode=render_mode)
    env.configure(
        {
            "observation": {
                "type": "Kinematics",
                # lat_off: 偏离车道中心的横向距离，=0时在正中间
                # ang_off: 车头相对车道方向的偏航角，弯道时自然增大（隐含弯道信息）
                # cos_h/sin_h: sin_h>0 car heading left, sin_h<0 heading right
                "features": ["lat_off", "ang_off", "vx", "vy", "cos_h", "sin_h"],
                "vehicles_count": 1,
                "absolute": False,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
            },
            "other_vehicles": 0,
            "action_reward": 0.0,  # 不惩罚转向动作，让agent自由学习转弯
            "lane_centering_cost": 8,  # 偏离中心惩罚更陡峭，梯度信号更强
            "lane_centering_reward": 1,
            "collision_reward": -5,  # 强出界惩罚，GAE向前传播"危险"信号
            "policy_frequency": 5,
            "simulation_frequency": 15,
            "duration": 300,
            "screen_width": 1600,
            "screen_height": 800,
            "scaling": 5,
        }
    )
    # Reset first to apply config, then wrap with FlattenObservation
    # so the wrapper sees the correct observation space
    env.reset()
    if hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        env.unwrapped.viewer.close()
        env.unwrapped.viewer = None
    env = FlattenObservation(env)
    return env
