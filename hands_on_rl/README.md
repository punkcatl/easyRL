# 动手学强化学习 — 章节练习

# Hands-on Reinforcement Learning — Chapter Exercises

教材网站 / Textbook: https://hrl.boyuai.com

GitHub: https://github.com/boyu-ai/Hands-on-RL

本目录存放跟教材练习的代码，按章节编号组织。

This directory contains exercises following the textbook, organized by chapter number.

---

## 必读章节 / Required Chapters

以下章节构成完整学习路径，与 `algorithms/` 对应：

```
基础篇 Part 1:
  02 多臂老虎机                       — 探索与利用的基本概念
  03 马尔可夫决策过程                   — RL 的数学框架
  04 动态规划算法                       — 策略迭代 / 价值迭代
  05 时序差分算法 (Sarsa / Q-Learning)  → algorithms/q_learning

进阶篇 Part 2:
  07 DQN 算法                          → algorithms/dqn
  09 策略梯度算法                       → algorithms/policy_gradient
  10 Actor-Critic 算法                  — PPO 前置知识，快读
  12 PPO 算法                          → algorithms/ppo
  13 SAC 算法                          → algorithms/sac
```

## 可跳过章节 / Optional Chapters

```
  01 初探强化学习                 — 概述，快速浏览即可
  06 Dyna-Q                      — model-based，本项目不涉及
  08 DQN 改进 (Double/Dueling/PER) — 锦上添花，非主线
  11 TRPO                        — PPO 的前身，理解 PPO 后回看即可
  14 DDPG                        — 连续控制替代方案，SAC 更优
  Part 3 前沿篇                   — 超出项目范围
```

## 目录结构 / Directory Structure

```
02_slot_machine/         — 多臂老虎机
03_markov_dp/            — 马尔可夫决策过程
04_dp/                   — 动态规划
05_temporal_difference/  — 时序差分
```
