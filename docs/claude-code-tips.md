# Claude Code Tips

## Aliases

Configured in `~/.bashrc` and `~/.zshrc`:

```bash
alias cc='claude --dangerously-skip-permissions'
```

Usage:

- `claude` — Normal mode, asks for permission before actions
- `cc` — Fully autonomous mode, skips all permission checks

**Note**: In `cc` mode the agent won't ask for confirmation. Suitable for trusted local projects. Make sure you have git as a safety net.

## Permission Modes

| Command | Description |
|---------|-------------|
| `claude` | Default, confirms each step |
| `claude --permission-mode acceptEdits` | Auto-approves file edits |
| `claude --permission-mode auto` | Auto-approves most operations |
| `cc` (alias) | Skips all permissions, fully autonomous |

## Status Line (HUD)

Bottom status bar is configured, showing model name, context usage, cost, effort level, rate limit, and git branch.

Config location: `statusLine` field in `~/.claude/settings.json`.

## Running External Commands in Claude Code

Use the `!` prefix:

```
! git log --oneline -5
! python train.py
```

---

# Claude Code 使用技巧

## 快捷别名

在 `~/.bashrc` 和 `~/.zshrc` 中配置了以下别名：

```bash
alias cc='claude --dangerously-skip-permissions'
```

使用方式：

- `claude` — 正常模式，操作前会确认权限
- `cc` — 全自动模式，跳过所有权限检查，agent 直接执行

**注意**：`cc` 模式下 agent 不会问你确认，适合本地可信项目。确保有 git 兜底。

## 权限模式一览

| 启动方式 | 说明 |
|----------|------|
| `claude` | 默认，每步确认 |
| `claude --permission-mode acceptEdits` | 文件编辑自动同意 |
| `claude --permission-mode auto` | 自动批准大多数操作 |
| `cc` (alias) | 完全跳过权限，全自动 |

## Status Line (HUD)

底部状态栏已配置，显示模型名称、上下文使用率、费用、effort 等级、rate limit、git 分支。

配置位置：`~/.claude/settings.json` 中的 `statusLine` 字段。

## 在 Claude Code 中运行外部命令

使用 `!` 前缀：

```
! git log --oneline -5
! python train.py
```
