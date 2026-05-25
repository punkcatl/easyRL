# Superpowers Skills Backup

Snapshot of `claude-plugins-official/superpowers` v5.1.0 skills, taken 2026-05-25.

## Purpose

Backup of the official superpowers plugin skills in case the plugin updates and you want to reference or restore this version.

## How to Restore

If the plugin updates and you want to use this version instead:

1. Copy the skill folder to `~/.claude/plugins/cache/claude-plugins-official/superpowers/<version>/skills/`
2. Or create a project-level skill in `.claude/skills/<skill-name>/SKILL.md`

## Skill Index

### Development Flow (in order)

| # | Skill | Directory | Description |
|---|-------|-----------|-------------|
| 1 | brainstorming | `brainstorming/` | Explore requirements → produce design spec |
| 2 | writing-plans | `writing-plans/` | Turn spec into step-by-step implementation plan |
| 3 | executing-plans | `executing-plans/` | Execute plan sequentially with review checkpoints |
| 3b | subagent-driven-development | `subagent-driven-development/` | Execute plan with parallel subagents |
| 3c | dispatching-parallel-agents | `dispatching-parallel-agents/` | Dispatch 2+ independent agents in parallel |

### Quality Assurance

| Skill | Directory | Description |
|-------|-----------|-------------|
| test-driven-development | `test-driven-development/` | Write tests before implementation |
| systematic-debugging | `systematic-debugging/` | Root cause analysis for bugs |
| verification-before-completion | `verification-before-completion/` | Run verification before claiming done |

### Code Review

| Skill | Directory | Description |
|-------|-----------|-------------|
| requesting-code-review | `requesting-code-review/` | Have Claude review your code |
| receiving-code-review | `receiving-code-review/` | Process review feedback correctly |

### Git & Completion

| Skill | Directory | Description |
|-------|-----------|-------------|
| using-git-worktrees | `using-git-worktrees/` | Manage isolated worktrees |
| finishing-a-development-branch | `finishing-a-development-branch/` | Guide merge/PR/cleanup |

### Meta

| Skill | Directory | Description |
|-------|-----------|-------------|
| using-superpowers | `using-superpowers/` | Entry point, decides which skill to invoke |
| writing-skills | `writing-skills/` | Create and edit skills |

## File Structure

Each skill directory contains:
- `SKILL.md` — Main skill definition (instructions for Claude)
- Additional `.md` files — Supporting prompts, templates, or references

## Version Info

- Plugin: `superpowers@claude-plugins-official`
- Version: `5.1.0`
- Snapshot date: 2026-05-25

---

# Superpowers Skills 备份

`claude-plugins-official/superpowers` v5.1.0 技能快照，拍摄于 2026-05-25。

## 用途

官方 superpowers 插件技能的备份，以防插件更新后需要参考或恢复当前版本。

## 如何恢复

如果插件更新后想使用此版本：

1. 将技能文件夹复制到 `~/.claude/plugins/cache/claude-plugins-official/superpowers/<version>/skills/`
2. 或在项目级别创建技能 `.claude/skills/<skill-name>/SKILL.md`

## 技能索引

### 开发流程（按顺序）

| # | 技能 | 目录 | 说明 |
|---|------|------|------|
| 1 | brainstorming | `brainstorming/` | 探索需求 → 输出设计方案 |
| 2 | writing-plans | `writing-plans/` | 将方案转为逐步实现计划 |
| 3 | executing-plans | `executing-plans/` | 按计划顺序执行，带 review 检查点 |
| 3b | subagent-driven-development | `subagent-driven-development/` | 多 subagent 并行执行计划 |
| 3c | dispatching-parallel-agents | `dispatching-parallel-agents/` | 派发 2+ 个独立 agent 并行工作 |

### 质量保证

| 技能 | 目录 | 说明 |
|------|------|------|
| test-driven-development | `test-driven-development/` | 先写测试再写实现 |
| systematic-debugging | `systematic-debugging/` | 系统化根因分析 |
| verification-before-completion | `verification-before-completion/` | 声称完成前先跑验证 |

### 代码审查

| 技能 | 目录 | 说明 |
|------|------|------|
| requesting-code-review | `requesting-code-review/` | 让 Claude 审查你的代码 |
| receiving-code-review | `receiving-code-review/` | 正确处理 review 反馈 |

### Git 与收尾

| 技能 | 目录 | 说明 |
|------|------|------|
| using-git-worktrees | `using-git-worktrees/` | 管理隔离的 worktree |
| finishing-a-development-branch | `finishing-a-development-branch/` | 引导 merge/PR/清理流程 |

### 元技能

| 技能 | 目录 | 说明 |
|------|------|------|
| using-superpowers | `using-superpowers/` | 入口，决定调用哪个技能 |
| writing-skills | `writing-skills/` | 创建和编辑技能 |

## 文件结构

每个技能目录包含：
- `SKILL.md` — 主技能定义（给 Claude 的指令）
- 其他 `.md` 文件 — 辅助 prompt、模板或参考资料

## 版本信息

- 插件：`superpowers@claude-plugins-official`
- 版本：`5.1.0`
- 快照日期：2026-05-25
