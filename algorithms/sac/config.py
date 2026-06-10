config = {
    "n_episodes": 200,      # 训练总轮数（SAC样本效率高，150轮即可收敛）
    "lr": 3e-4,             # Adam学习率
    "gamma": 0.99,          # 折扣因子
    "tau": 0.005,           # target网络软更新系数
    "alpha": 0.2,           # 初始熵系数
    "auto_alpha": True,     # 自动调节熵系数
    "buffer_size": 100000,  # replay buffer容量
    "batch_size": 256,      # mini-batch大小
    "hidden_dim": 64,       # 隐藏层宽度，6维输入1维输出
    "start_steps": 500,     # 纯随机探索步数（填充buffer）
}
