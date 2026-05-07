# Sarsa vs Q-Learning

## Formula

```
Sarsa:      Q(s,a) ← Q(s,a) + α × [r + γ × Q(s', a')     - Q(s,a)]
Q-Learning: Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s', a') - Q(s,a)]
```

## Key Differences

```
                    Sarsa                           Q-Learning
─────────────────────────────────────────────────────────────────────────────
Next-step Q        Q(s', a')                       max Q(s', a')
                    a' chosen by ε-greedy           take max over all actions

Nature of a'       A decision                      A computation
                    must be produced, will           byproduct of max,
                    actually be executed next        discarded after calculation

Data requirement   5-tuple (s, a, r, s', a')       4-tuple (s, a, r, s')
                    a' passed in from outside        .max() computed internally

What it evaluates  Current ε-greedy policy          Optimal greedy policy
                    with exploration noise           no noise

Type               on-policy                        off-policy

Learned behavior   Conservative (avoid danger)      Aggressive (shortest path)
                    accounts for mistake cost        assumes no future mistakes
─────────────────────────────────────────────────────────────────────────────
```

## One-liner

Sarsa evaluates "how I'll actually do"; Q-Learning evaluates "how well I could possibly do".

---

# Sarsa vs Q-Learning（中文版）

## 公式

```
Sarsa:      Q(s,a) ← Q(s,a) + α × [r + γ × Q(s', a')     - Q(s,a)]
Q-Learning: Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s', a') - Q(s,a)]
```

## 核心区别

```
                    Sarsa                           Q-Learning
─────────────────────────────────────────────────────────────────────────────
下一步 Q 值        Q(s', a')                       max Q(s', a')
                    a' 是 ε-greedy 选出的            遍历所有动作取最大值

a' 的性质          一次决策                          一次计算
                    要主动产生，下一步真的执行它        max 的副产品，算完就扔

数据需求           五元组 (s, a, r, s', a')          四元组 (s, a, r, s')
                    a' 要从外部传入                   函数内部 .max() 搞定

评估的是谁         当前 ε-greedy 策略                 最优贪心策略
                    带探索噪声                        无噪声

类型               on-policy                        off-policy

学到的行为         保守（远离危险）                    激进（最短路径）
                    把犯错成本算进去                   假设以后不犯错
─────────────────────────────────────────────────────────────────────────────
```

## 一句话

Sarsa 评估"我实际会怎样"，Q-Learning 评估"我最好能怎样"。
