import numpy as np
import matplotlib.pyplot as plt

'''1.随机生成K个老虎机中奖概率,并实现拉动某个老虎机杆以及获得奖励'''
class BernoulliBandit:
    def __init__(self, K):
        self.probs = np.random.uniform(size=K) #随机生成K个概率值[0,1)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K
        
    def step(self, k):
        if np.random.rand() < self.probs[k]: #随机生成的概率小于中奖概率即认为等价于中奖
            return 1.0
        else:
            return 0.0

np.random.seed(1) #设定随机种子，使可重复
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成 %d 臂老虎机" %K)
print("获奖概率最大的拉杆为 %d 号 , 获奖概率为 %.4f" %(bandit_10_arm.best_idx, bandit_10_arm.best_prob))


'''2.多臂老虎机算法框架,包括记录操作、进行次数、单次/总误差等信息，按给定次数运行老虎机，选杆策略（由子类实现）'''
class Solver: 
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0. #累积误差
        self.actions = [] #记录每一步的动作
        self.regrets = [] #记录每一步的误差

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
        
    def run_one_step(self):
        raise NotImplementedError #选哪根杆的策略,由子类实现
    
    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

'''3.实现epsilon-贪婪算法，实现选杆策略，并运行算法，输出累积懊悔值及绘图'''         
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit) #super语法可以调用父类的方法，super(EpsilonGreedy)表示调用EpsilonGreedy的父类Solver
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K) #初始化每个拉杆的估计获奖概率为init_prob
    
    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) #随机选择一个拉杆
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k) #执行动作，获得奖励
         
        '''#增量式更新估计的获奖概率(均值)
        mean_k = 1/k*(r1+r2+...+rk) 
               = 1/k*(rk+ r1+r2+...+rk-1)
               = 1/k*(rk+ (k-1)*[mean_k-1]) 
               = mean_k-1 + 1/k*(rk - mean_k-1) 
        '''
        self.estimates[k] += 1./(self.counts[k] + 1) * (r - self.estimates[k]) 
        return k
    
'''3.1 epsilon随时间衰减的贪婪算法'''
class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0
    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count: #epsilon随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
            
        r = self.bandit.step(k)
        self.estimates[k] += 1./(self.counts[k] + 1) * (r - self.estimates[k]) 
        return k
    
'''3.2 上置信界算法，考虑不确定性因素选择拉杆。不确定性随样本数增加而减小，不确定性越大越值得探索'''
'''霍夫丁不等式
真实期望E[X]超过样本均值x_bar一个u的概率,不超过e^(-2nu^2)
这个不等式说明了样本均值偏离真实期望的概率随样本数增加而指数衰减'''
class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef
    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + np.sqrt(np.log(self.total_count) / (2*(self.counts + 1))) #计算上置信界
        k = np.argmax(ucb) #选择上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1./(self.counts[k] + 1) * (r - self.estimates[k]) 
        return k
    
def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers): #enumerate会按该顺序生成(index, element)对
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    # plt.yticks(np.arange(0,6,step=0.1))
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

# '''4.绘制epsilon=0.01的累积误差'''
# np.random.seed(1)
# epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
# epsilon_greedy_solver.run(5000)
# print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
# # print('epsilon_贪婪算法的估计获奖概率: ', epsilon_greedy_solver.estimates)
# # print('真实获奖概率: ', bandit_10_arm.probs)
# # print('选择的拉杆次数: ', epsilon_greedy_solver.counts)
# plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])  

# '''5.绘制不同epsilon值下的累积误差'''
# np.random.seed(0)
# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
# epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons] #.format()是格式化输出
# for solver in epsilon_greedy_solver_list:
#     solver.run(5000)
# plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

# '''5.1 绘制衰减epsilon下的累积误差'''
# np.random.seed(1)
# decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
# decaying_epsilon_greedy_solver.run(5000)
# print('衰减epsilon-贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
# plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

# '''5.2 绘制上置信界算法的累积误差'''
# np.random.seed(1)
# coef = 1  # 控制不确定性比重的系数
# UCB_solver = UCB(bandit_10_arm, coef)
# UCB_solver.run(5000)
# print('上置信界算法的累积懊悔为：', UCB_solver.regret)
# plot_results([UCB_solver], ["UCB"])

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self.successes = np.ones(self.bandit.K)  # 每个拉杆的成功(奖励为1)次数
        self.failures = np.ones(self.bandit.K)   # 每个拉杆的失败(奖励为0)次数

    def run_one_step(self):
        #按Beta(α, β)分布抽取随机数，区间 (0,1) 之间，Beta 分布常用于伯努利/二项概率的贝叶斯后验建模
        '''
        Beta分布的数学基础:
        Beta分布是定义在[0,1]区间上的连续概率分布，特别适合对概率和比例进行建模。
        #核心参数
        Beta分布由两个形状参数控制：
        α (alpha)：可理解为观测到的成功次数加1
        β (beta)：可理解为观测到的失败次数加1
        这两个参数决定了分布的形状特征：
        α值增大会使分布向1偏移，表示成功概率增加
        β值增大会使分布向0偏移，表示失败概率增加
        #分布特征
        Beta分布的形状由α和β的相对大小决定：
        α > β：分布偏向1，表示成功概率较高
        β > α：分布偏向0，表示失败概率较高
        α = β：分布关于0.5对称
        '''
        #对数组的每一对(α_i, β_i)返回一个对应的采样值，结果 samples 是形状与 self.successes/self.failures 相同的一维数组。也就是说，返回的是每个臂的一次独立 Beta 抽样。
        samples = np.random.beta(self.successes, self.failures) 
        k = np.argmax(samples)  # 选择采样值最大的拉杆
        r = self.bandit.step(k)

        # 更新成功和失败次数
        if r == 1.0:
            self.successes[k] += 1
        else:
            self.failures[k] += 1
            
        return k


np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])