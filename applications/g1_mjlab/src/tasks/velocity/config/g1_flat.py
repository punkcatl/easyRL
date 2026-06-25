from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.env_cfg import make_g1_flat_env_cfg
from src.tasks.velocity.rl_cfg import make_g1_ppo_runner_cfg
from mjlab.rl import MjlabOnPolicyRunner

register_mjlab_task(
    task_id="G1-Flat-v0",
    env_cfg=make_g1_flat_env_cfg(num_envs=2048),
    play_env_cfg=make_g1_flat_env_cfg(num_envs=16, play=True),
    rl_cfg=make_g1_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
