import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0])) #更新x坐标，并限制下个状态的x坐标在边界内
                    next_y = min(self.nrow - 1, max(0, i + change[a][1])) #更新y坐标，并限制下个状态的y坐标在边界内
                    next_state = next_y * self.ncol + next_x #拍扁成一维数组位置
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置不在终点(即在悬崖!)
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

    def step(self, action):
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done
    

class PolicyIteration:
    '''策略迭代算法'''
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow #初始化价值为0，每个格子一个0
        #pi即pi[a|s],策略pi在状态s下选择动作a的概率
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)] #初始化为均匀随机策略，每个格子4个动作的概率均为0.25
        self.theta = theta #策略评估收敛阈值
        self.gamma = gamma #折扣因子
        
    def policy_evaluation(self): #策略评估
        cnt = 1 #计数器
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow): #对于每个格子里的状态s
                qsa_list = [] #计算状态s下所有的Q(s,a)价值
                for a in range(4): #对于每个动作a
                    qsa = 0
                    for res in self.env.P[s][a]: #[!]这里是更通用的写法，P[s][a]可能是个列表，有多种不同的转移可能比如P[s][a] = [(0.8, s1, r1, False),(0.2, s2, r2, False),],实际上当前环境只有一种转移可能，循环只执行1次
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1-done)) #p:本迷宫概率为1， 1-done：如果done==true代表完成了则价值为0
                        #本章环境比较特殊，奖励和下一个状态有关，所以需要和状态转移概率相乘
                    qsa_list.append(self.pi[s][a] * qsa) #将动作期望回报Q(s,a)按策略概率pi(a|s)加权，状态价值：Vπ(s) = Σ_a π(a|s) Q(s,a)，后面要进行求和来得到状态价值Vπ(s)
                new_v[s] = sum(qsa_list) #状态价值函数Vπ(s) = Σ_a π(a|s) Q(s,a)，所以要对所有的pi(a|s)*qsa求和才能得到
                max_diff = max(abs(new_v[s] - self.v[s]), max_diff)
            self.v = new_v
            if max_diff < self.theta: break #满足收敛条件，退出评估迭代
            cnt += 1
        print("策略评估进行%d轮后完成" % cnt)
    
    def policy_improvement(self): #策略提升
        for s in range(self.env.nrow * self.env.ncol): #遍历每个格子中的状态s
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list) #找出最大的动作价值期望max Q(s,a)
            cntq = qsa_list.count(maxq) #计算有几个动作得到了最大的Q值
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list] #让同为max值的这些动作均分概率，其他动作置为0(贪心策略！)
        print("策略提升完成")
        return self.pi #该策略pi包含每个状态s下4个动作的执行概率
    
    def policy_iteration(self): #策略迭代
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break
    
#now, 环境和策略迭代的代码已完成，现在编写一个方便观察的打印代码
def print_agent(agent, env_or_meaning, action_meaning=None, disaster=[], end=[]):
    if action_meaning is None:
        # 旧的调用方式: print_agent(agent, action_meaning, disaster, end)
        env = agent.env
        action_meaning = env_or_meaning
    else:
        # 新的调用方式: print_agent(agent, env, action_meaning, disaster, end)
        env = env_or_meaning

    print("状态价值：")
    for i in range(env.nrow):
        for j in range(env.ncol):
            if hasattr(agent, 'v'):
                print('%6.6s' % ('%.3f' % agent.v[i * env.ncol + j]), end=' ')
            elif hasattr(agent, 'Q_table'):
                print('%6.6s' % ('%.3f' % agent.Q_table[i * env.ncol + j].max()), end=' ')
            else:
                print('%6.6s' % ('%.3f' % 0), end=' ')
        print()
    print("策略： ")
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end = ' ')
            else:
                if hasattr(agent, 'pi'):
                    a = agent.pi[i * env.ncol + j]
                else:
                    a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

#1.策略迭代    
# env = CliffWalkingEnv()
# action_meaning = ['^', 'v', '<', '>']
# theta = 0.001
# gamma = 0.9
# agent = PolicyIteration(env, theta, gamma)
# agent.policy_iteration()
# print_agent(agent, action_meaning, disaster = list(range(37,47)), end = [47])   #第38到47(对应[37]～[46])个点是悬崖，第48(对应[47])个点是终点。range(a,b)左闭右开                     
                    
                        
class ValueIteration:
    """价值迭代算法"""
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow #初始化状态为0
        self.theta = theta #价值收敛阈值
        self.gamma = gamma
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)] #None表示未初始化的空值
        
    def value_iteration(self):
        cnt = 0
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = [] #计算s状态下所有的Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1- done))
                    qsa_list.append(qsa) #这行和下一行是价值迭代和策略迭代的主要区别
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break #满足收敛条件，推出迭代]
            cnt += 1
        print("价值迭代一共进行了%d轮" % cnt)
        self.get_policy()
        
    def get_policy(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
                maxq = max(qsa_list)
                cntq = qsa_list.count(maxq)
                self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]

# #2.价值迭代
# env = CliffWalkingEnv()
# action_meaning = ['^', 'v', '<', '>']
# theta = 0.001
# gamma = 0.9
# agent = ValueIteration(env, theta, gamma)
# agent.value_iteration()
# print_agent(agent, action_meaning, disaster = list(range(37,47)), end = [47])
                

if __name__ == "__main__":
    #冰湖环境
    # import gym
    import gymnasium as gym
    env = gym.make("FrozenLake-v1", render_mode = "rgb_array") #创建环境，4x4网格，起点S,终点G,地面F(Frozen，安全但滑),冰洞H
    env = env.unwrapped # 解封装才能获取原始环境对象，访问内部属性。比如访问状态转移矩阵P
    obs, info = env.reset()

    holes = set() #创建空的集合
    ends = set()
    for s in env.P: #env.P[state][action] = [(probability, next_state, reward, done), ...]
        for a in env.P[s]:
            for s_ in env.P[s][a]: #对于每个状态 s 和动作 a，可能有多个 (p, s', r, done) 转移（因为 FrozenLake 是随机环境：想往右走，可能因冰面打滑而走向其他方向）。
                if s_[2] == 1.0: # 如果奖励为1.0(即目标，只有目标点是1.0奖励，其他格子奖励均为0)
                    ends.add(s_[1]) # 把目标增加到ends数组中
                if s_[3] == True: #s_[3]是done标志位，掉到洞里或者到达终点都是True
                    holes.add(s_[1]) #此时holes包含冰洞位置和终点位置
    holes = holes - ends #剔除终点位置后，holes数组只包含冰洞位置
    print("冰洞的索引：", holes)
    print("目标的索引：", ends)

    # for a in env.P[14]: #查看目标左边一格的状态转移信息(probability, next_state, reward, done)   4x4棋盘，1～16格子，编号0～15,目标为编号15,目标左侧一格编号为14
    #       print(env.P[14][a])

    # #3.在冰湖环境测试策略迭代算法
    # action_meaning = ['<', 'v', '>', '^']
    # theta = 1e-5
    # gamma = 0.9
    # agent = PolicyIteration(env, theta, gamma)
    # agent.policy_iteration()
    # print_agent(agent, action_meaning, disaster = [11, 12, 5, 7], end = [15])

    #4.在冰湖环境测试价值迭代算法
    action_meaning = ['<', 'v', '>', '^']
    theta = 1e-5
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, disaster = [11, 12, 5, 7], end = [15])







   
    #---显示地图---#
    frame = env.render()   # 返回 ndarray
    import matplotlib.pyplot as plt
    plt.imshow(frame)
    plt.axis('off')
    plt.show()
    env.close()
    #---显示地图---#
      
