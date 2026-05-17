import numpy as np
np.random.seed(0)
#状态转移矩阵P
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0]  #奖励函数，针对不同状态
gamma = 0.5 #折扣因子

def compute_return(start_index, chain, gamma):
    # 反向递推
    G = 0
    for i in reversed(range(start_index, len(chain))):  #range(0,4)左闭右开，得到的整数序列是0,1,2,3。reversed后得到3,2,1,0
        G = gamma * G + rewards[chain[i]-1]
    #验证计算结果是否正确，使用显式求和的方式进行对比
    # G = sum((gamma**k)*rewards[chain[k]-1] for k in range(len(chain)))
    
    return G

#一个状态序列， s1-s2-s3-s6
chain = [1, 2, 3, 6] #如果状态编号从0开始，对应0,1,2,5
start_index = 0
G = compute_return(start_index, chain, gamma)
# print("根据本序列计算得到的回报G为: %s." %G)

#实现求解价值函数的解析解方法，并据此计算该马尔可夫奖励过程中所有状态的价值
def compute(P, rewards, gamma, states_num):
    rewards = np.array(rewards).reshape(-1, 1)  #转化为列向量，reshape()中的-1表示自动推断维度大小，1表示转换为每行只有1列。reshape(-1,1)即表示变成一列，行数自动匹配元素总数
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards) # V = (I-γP)^-1 * R
    return value

V = compute(P, rewards, gamma, states_num=6)
# print("MRP中每个状态的价值分别为:\n", V)



S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子

MDP = (S, A, P, R, gamma)

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2

gamma = 0.5
# 转化后的MRP的状态转移矩阵
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
# print("MDP中每个状态价值分别为\n", V)



def sample(MDP, Pi, timestep_max, number):
    S, A, P, R, gamma = MDP
    episodes = [] #所有序列的集合
    for _ in range(number):
        episode = [] #一个序列
        timestep = 0
        s = S[np.random.randint(4)] #随机选择初始状态，randint(a,b)左闭右开，randint(a)则默认下限为0,上限为a
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            #1.在状态s下根据策略选择动作a
            rand, temp = np.random.rand(), 0
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0) #.get(a,0)读取a键对应的值，若a键不存在则返回0(第二个参数为异常时的默认值)。另外temp需要累加的原因：在状态s下选择动作a1,a2,...的概率，选择这些的动作的概率加起来应为1。相当于把 [0,1) 区间按概率切分成几段，随机落点在哪一段，就选哪个动作。
                if rand < temp:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            
            #2.确定好(s,a)后根据状态转移概率P得到下一个状态s_next
            rand, temp = np.random.rand(), 0
            for s_opt in S:
                temp += P.get(join(join(s,a), s_opt), 0)
                if rand < temp:
                    s_next = s_opt
                    break
            episode.append((s,a,r,s_next))
            s = s_next #更新当前状态为s_next，开始接下来的循环
        episodes.append(episode)
    return episodes

#采样5次，每个序列最长不超过20步
episodes = sample(MDP, Pi_1, timestep_max=20, number=5)
# print('第一条序列\n', episodes[0])
# print('第二条序列\n', episodes[1])
# print('第五条序列\n', episodes[4])

#对所有采样序列计算所有状态的价值
def MC(episodes, V, N, gamma):
    for episode in episodes: #遍历所有序列
        G = 0
        for i in range(len(episode)-1, -1, -1): #反向遍历单个序列，步长为1
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] += 1
            V[s] = V[s] + (G - V[s])/N[s]

timestep_max = 20
#采样1000次
episodes = sample(MDP, Pi_1, timestep_max, number=1000)
gamma = 0.5
V = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0} #记录每个状态的价值
N = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0} #记录每个状态被访问

MC(episodes, V, N, gamma)
# print('使用蒙特卡洛方法计算MDP的状态价值:\n', V)



def occupancy(episodes, s, a, timestep_max, gamma):
    ''' 计算状态动作对(s,a)出现的频率,以此来估算策略的占用度量 '''
    rho = 0
    total_times = np.zeros(timestep_max)  # 记录所有episodes累计 经历每个时间步的次数
    occur_times = np.zeros(timestep_max)  # 记录(s_t,a_t)=(s,a)的次数
    for episode in episodes: #遍历所有序列
        for i in range(len(episode)): #遍历单个序列的每一个(s,a,r,s_next)组合
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)): #倒序遍历 999,998,...,1,0
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


gamma = 0.5
timestep_max = 1000

episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
print(rho_1, rho_2)
            
            
        
                    
