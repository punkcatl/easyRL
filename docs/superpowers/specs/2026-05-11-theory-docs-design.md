# Theory Documentation Design

## Overview

Add a `docs/theory.md` file to each algorithm folder, providing bilingual (English + Chinese) theory explanations with formula-to-code mappings.

## File Location

```
algorithms/<algo>/docs/theory.md
```

Applies to: q_learning, dqn, policy_gradient, ppo, sac (5 files total).

## Document Structure

Each `theory.md` follows this template:

```
# Algorithm Name

## Intuition
One paragraph explaining what the algorithm does and why it works.

## Core Formula
1-3 key formulas in LaTeX.

## Formula-to-Code Mapping
Table or list pointing specific lines in agent.py to formulas.

## Deep Dive (Optional)
Full mathematical derivation for interested readers.

---

# 算法名称（中文版）

Same four sections in Chinese.
```

## Style Guidelines

- Target audience: both RL newcomers (with programming + basic math) and experienced practitioners needing a refresher
- Intuition section: no formulas, plain language, build mental model first
- Core Formula: only the essential update rules, not derivations
- Code Mapping: reference specific line numbers in `agent.py`, use format `agent.py:L42`
- Deep Dive: full derivation, can assume calculus and probability knowledge
- LaTeX: use GitHub-compatible `$...$` inline and `$$...$$` block format

## Scope

- 5 theory documents (one per algorithm)
- No changes to existing code files
- Update README to mention the theory docs

---

# 理论文档设计（中文版）

## 概述

在每个算法文件夹中添加 `docs/theory.md` 文件，提供中英双语理论解释与公式-代码映射。

## 文件位置

```
algorithms/<algo>/docs/theory.md
```

适用于：q_learning、dqn、policy_gradient、ppo、sac（共 5 个文件）。

## 文档结构

每个 `theory.md` 遵循以下模板：

```
# 算法名称

## 直觉
一段话解释算法做什么以及为什么有效。

## 核心公式
1-3 个关键公式。

## 公式与代码对应
表格或列表，将 agent.py 中的具体行号指向公式。

## 深入推导（选读）
完整数学推导，面向感兴趣的读者。

---

# Algorithm Name (English)

同样四个章节的英文版。
```

## 风格指南

- 目标读者：RL 新手（有编程 + 基础数学背景）和需要复习的有经验从业者
- 直觉章节：不用公式，用大白话，先建立心智模型
- 核心公式：仅包含核心更新规则，不展开推导
- 代码对应：引用 `agent.py` 中的具体行号，格式为 `agent.py:L42`
- 深入推导：完整推导，可假设读者具备微积分和概率论知识
- 公式格式：使用 GitHub 兼容的 `$...$` 行内和 `$$...$$` 块格式

## 范围

- 5 个理论文档（每个算法一个）
- 不修改现有代码文件
- 更新 README 中提及理论文档
