# AI Engineering Layers: Prompt, Skill, and Harness

AI engineering isn't just "writing prompts." It's a layered discipline. Here's how the layers relate, why each exists, and when to invest in which.

## The Three Core Layers

**Analogy: cooking.**

| Layer | What it is | Cooking analogy |
|-------|-----------|-----------------|
| Prompt Engineering | Crafting a single input to get the desired output | Telling the chef "make tomato egg, not spicy, extra sugar" |
| Skill Engineering | Packaging a proven workflow into a reusable recipe | Writing a recipe card with steps, timing, and ratios |
| Harness Engineering | Building the runtime environment, capabilities, and constraints | Designing the entire kitchen: stoves, tools, storage, safety rules |

## "What Does Harness Actually Do?"

A single recipe (skill) can get one dish done. But run a restaurant and you need more:

| You see this... | Harness provides it |
|-----------------|-------------------|
| AI remembers your preferences across sessions | Memory system (file-based persistence + auto-loading) |
| AI dispatches multiple sub-agents in parallel | Process scheduling and coordination |
| AI can't delete files without your approval | Permission boundaries enforced in code |
| Lint runs automatically before every commit | Hooks (event-triggered actions) |
| AI can read files, run commands, search the web | Tool registration and execution layer |

## "Can't I Just Write All This in a Markdown File?"

No. The key distinction:

- **Skill (written in md)** = strong guidance. AI follows it most of the time, but compliance is not guaranteed by the system. Soft constraint.
- **Harness (code + config)** = enforcement. The system blocks or allows actions regardless of what AI "wants." Hard constraint.

| Written in md (skill) | Implemented in harness |
|-----------------------|-----------------------|
| "Please ask before deleting files" | Tool layer intercepts deletion — blocked without approval |
| "Remember user preferences" | Disk read/write + auto-load mechanism — actually persists |
| "You can run tasks in parallel" | Process manager spawns real concurrent agents |

A sign on the wall says "no stealing." A lock on the door *prevents* it. Skill is the sign; harness is the lock.

## The Broader Landscape

Beyond prompt/skill/harness, the field includes:

| Concept | Focus | Analogy |
|---------|-------|---------|
| Context Engineering | What goes into the context window, compression, eviction | How to arrange ingredients on a tiny counter |
| Agent Engineering | Multi-step autonomous loops: plan → act → reflect | Chef reads the menu, decides order, judges doneness |
| Eval Engineering | Measuring output quality | Taste testers + scoring rubrics |
| RAG Engineering | Retrieving relevant external knowledge for the model | Quickly finding the right page in a reference book |
| Guardrail Engineering | Output filtering, safety fences | Toxic dishes can't leave the kitchen, no matter what |
| Fine-tuning | Modifying model weights directly | Changing the chef's instincts, not giving instructions |

**How they relate:**

```
┌───────────────────────────────────────────────────────┐
│  Fine-tuning (orthogonal — changes the model          │
│  itself, independent of runtime setup)                │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│  Harness (runtime environment)                        │
│    ├── Context Engineering (input management)         │
│    │     ├── RAG* (external knowledge retrieval)      │
│    │     └── Skill (workflow recipes)                 │
│    │           └── Prompt (single instruction)        │
│    ├── Agent Engineering** (autonomous loops)         │
│    ├── Guardrail (output constraints)                 │
│    └── Eval (quality measurement)                     │
└───────────────────────────────────────────────────────┘

*  RAG's purpose is filling context, but its implementation
   (vector DBs, chunking, indexing) is harness-level infra.
** Agent loops consume context engineering and use skills/prompts
   internally. In many architectures, the agent IS the orchestration
   layer — shown as peer here for simplicity.
```

## When to Invest in Which Layer

| Your situation | Focus on |
|---------------|----------|
| Just starting with AI tools | Prompt — learn to ask well |
| Repeating the same type of task | Skill — write it down once, reuse |
| Team collaboration / need safety boundaries | Harness — enforce rules in code |
| Task requires multi-step autonomy | Agent — design plan-act-reflect loops |
| Output quality is unpredictable | Eval — measure before you iterate |
| Model doesn't know domain-specific facts | RAG or fine-tuning (different trade-offs: retrieval at runtime vs. baking knowledge into weights) |

## One-Line Summary

Prompt is what you say; skill is how you work; harness is where you run.

---

# AI 工程分层：Prompt、Skill 与 Harness

AI 工程不只是"写提示词"。它是一个分层体系。这里梳理各层之间的关系、存在原因，以及什么时候该投资哪一层。

## 三个核心层次

**类比：做菜。**

| 层次 | 是什么 | 做菜类比 |
|------|--------|----------|
| Prompt Engineering | 精心设计单次输入，获得期望输出 | 对厨师说"做个番茄炒蛋，不辣，多放糖" |
| Skill Engineering | 将验证过的工作流封装成可复用流程 | 写一张菜谱卡片：步骤、火候、调料比例 |
| Harness Engineering | 构建运行环境、能力边界和约束机制 | 设计整个厨房：灶台、工具、储存方式、安全规则 |

## "Harness 到底有什么用？"

一张菜谱（skill）能搞定一道菜。但如果你开的是餐厅，就需要更多：

| 你看到的现象 | 背后是 harness 在提供 |
|-------------|---------------------|
| AI 跨会话记住你的偏好 | 记忆系统（文件持久化 + 自动加载） |
| AI 同时派出多个子代理并行工作 | 进程调度与协调 |
| AI 不能在没有你确认的情况下删文件 | 代码层面的权限边界 |
| 每次提交前自动跑 lint | Hooks（事件触发的动作） |
| AI 能读文件、跑命令、搜网页 | 工具注册与执行层 |

## "写在 md 文件里不就行了？"

不行。关键区别：

- **Skill（写在 md 里）**= 强引导。AI 大部分时候会遵循，但系统层面不保证执行。软约束。
- **Harness（代码 + 配置）**= 强制。系统直接拦截或放行，与 AI "想不想"无关。硬约束。

| 写在 md 里（skill） | 由 harness 实现 |
|--------------------|----------------|
| "删文件前请先问用户" | 工具层拦截删除操作——不批准就执行不了 |
| "记住用户偏好" | 磁盘读写 + 自动加载机制——真的能持久化 |
| "可以并行执行任务" | 进程管理器启动真正的并发代理 |

墙上贴个告示写"请勿偷窃"。门上装把锁才能**防住**。Skill 是告示，harness 是锁。

## 更广的全景

prompt/skill/harness 之外，这个领域还有：

| 概念 | 关注什么 | 类比 |
|------|---------|------|
| Context Engineering | 上下文窗口里放什么、怎么压缩、什么时候丢弃 | 工作台就这么大，食材怎么摆 |
| Agent Engineering | 多步骤自主循环：规划→执行→反思 | 厨师自己看菜单、决定顺序、判断火候 |
| Eval Engineering | 衡量输出质量 | 试吃员 + 评分标准 |
| RAG Engineering | 从外部知识库检索相关信息给模型 | 快速翻到参考手册正确的那一页 |
| Guardrail Engineering | 输出过滤、安全围栏 | 有毒的菜不能出厨房门 |
| Fine-tuning | 直接修改模型权重 | 不是教厨师按菜谱做，而是改变厨师本人的口味直觉 |

**层次关系：**

```
┌───────────────────────────────────────────────────────┐
│  Fine-tuning                                          │
│  (orthogonal: changes the model itself,               │
│   independent of runtime setup)                       │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│  Harness (runtime environment)                        │
│    ├── Context Engineering (input management)         │
│    │     ├── RAG* (external knowledge retrieval)      │
│    │     └── Skill (workflow recipes)                 │
│    │           └── Prompt (single instruction)        │
│    ├── Agent Engineering** (autonomous loops)         │
│    ├── Guardrail (output constraints)                 │
│    └── Eval (quality measurement)                     │
└───────────────────────────────────────────────────────┘

*  RAG 的目的是填充上下文，但其实现（向量数据库、分块、索引）
   属于 harness 层基础设施。
** Agent 循环内部会消费上下文管理、使用 skill/prompt。
   在很多架构中，agent 本身就是编排层——为简洁放在同级。
```

## 什么时候该投资哪一层

| 你的情况 | 该关注 |
|---------|--------|
| 刚开始用 AI 工具 | Prompt——学会怎么问 |
| 反复做同类任务 | Skill——写一次，反复用 |
| 团队协作 / 需要安全边界 | Harness——用代码强制规则 |
| 任务需要多步自主决策 | Agent——设计规划→执行→反思循环 |
| 输出质量不稳定 | Eval——先度量再迭代 |
| 模型不了解领域知识 | RAG 或 fine-tuning（取舍不同：运行时检索 vs. 将知识烧进权重） |

## 一句话总结

Prompt 是你说的话，Skill 是你做事的方式，Harness 是你运行的环境。

