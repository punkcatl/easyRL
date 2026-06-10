config = {
    "n_episodes": 5000,     # 训练总轮数（目标200k+总步数）
    "lr": 3e-4,             # Adam学习率（Critic自动设为3倍）
    "gamma": 0.99,          # 折扣因子，越接近1越重视长期奖励
    "gae_lambda": 0.95,     # GAE的lambda，平衡优势估计的偏差与方差
    "clip_eps": 0.2,        # PPO clip范围，限制新旧策略比值在[0.8, 1.2]内
    "entropy_coef": 0.005,  # 熵奖励系数
    "epochs": 10,           # 每次收集数据后重复训练的轮数
    "batch_size": 64,       # mini-batch大小
    "hidden_dim": 64,       # 隐藏层宽度，6维输入1维输出
    "rollout_steps": 2048,  # rollout buffer大小，累积足够步数再更新
}
