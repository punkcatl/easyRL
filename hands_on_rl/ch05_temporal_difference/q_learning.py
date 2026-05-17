import numpy as np
from hands_on_rl.ch04_dp.DP import CliffWalkingEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
from hands_on_rl.ch04_dp.DP import print_agent

class QLearning:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action]) #初始化Q(s,a)表格
        self.n_action = n_action #动作个数
        self.alpha = alpha #学习率
        self.gamma = gamma #折扣因子
        self.epsilon = epsilon #探索率
        
    def take_action(self, state): #选取下一步的操作
        if np.random.random() < self.epsilon: #以epsilon的概率随机选取一个动作
            action = np.random.choice(self.n_action)
        else:
            action = np.argmax(self.Q_table[state]) #以1-epsilon的概率选取Q值最大的动作
        return action
    
    def best_action(self, state): #用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i] == Q_max: #如果Q值等于最大值，则将该动作标记为1
                a[i] = 1
        return a
    
    def update(self, s0, a0, r, s1): #更新Q值
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

     
np.random.seed(0)
env = CliffWalkingEnv()
ncol, nrow = env.ncol, env.nrow
epsilon = 0.1 #探索率
alpha = 0.1 #学习率
gamma = 0.9 #折扣因子
agent = QLearning(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 1500 #训练500轮，智能体在环境中运行的序列的数量

return_list = [] #记录每一条序列的回报
for i in range(10): #显示10个进度条
    # tqdm是一个Python库，用于显示循环的进度条，提供了一个可视化的方式来跟踪循环的执行进度和剩余时间。
    print()  #在每个进度条之间加一个空行
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)): # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                episode_return += reward #这里回报的计算不进行折扣因子衰减,这里是每一轮的总回报
                agent.update(state, action, reward, next_state)
                state = next_state
            return_list.append(episode_return)
            if(i_episode + 1) % 10 == 0: #每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:]) #取最后10个元素算平均值
                })
            pbar.update(1) #让进度条前进1步

action_meaning = ['^','v','<','>'] #定义动作的含义，分别是上、下、左、右
print('Q-learning算法最终收敛得到的策略为:')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

print('\nQ表 (每个状态下4个动作的Q值，动作顺序: ^上 v下 <左 >右):')
print('-' * 70)
for i in range(env.nrow):
    for j in range(env.ncol):
        s = i * env.ncol + j
        q_values = agent.Q_table[s]
        best = np.argmax(q_values)
        print(f'状态{s:2d}({i},{j:2d}): ^{q_values[0]:7.3f} v{q_values[1]:7.3f} <{q_values[2]:7.3f} >{q_values[3]:7.3f}  最优:{action_meaning[best]}')
    print('-' * 70)

episode_list = list(range(len(return_list))) #生成一个从0到len(return_list)-1的整数列表，这里是[0, 1, 2..., 499]
plt.plot(episode_list, return_list) #绘制回报随序列变化的曲线图
plt.xlabel('Episode') #设置x轴标签
plt.ylabel('Returns') #设置y轴标签
plt.title('Q Learning on {}'.format('Cliff Walking')) #设置图表标题
plt.show() #显示图表


                
        
    

