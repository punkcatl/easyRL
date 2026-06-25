"""PPO runner config for G1 humanoid — larger network for 15 DOF."""
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg


def make_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=1e-3,
            schedule="adaptive",
            desired_kl=0.01,
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.01,
            clip_param=0.2,
            max_grad_norm=1.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            value_loss_coef=1.0,
        ),
        experiment_name="g1_mjlab_teacher",
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=1500,
    )
