# Agent Tooling

This project uses three complementary systems to support AI-assisted research workflows. Here's what each does and when it's used.

## Overview

| System | Location | Purpose | How it works |
|--------|----------|---------|--------------|
| **Skills** | `skills/` | Give agents domain knowledge | Auto-discovered at startup, activated when the task matches |
| **Slash Commands** | `.claude/commands/` | Explicit user-triggered workflows | User types `/command`, agent follows step-by-step |
| **OpenSpec** | `openspec/` | Change management for big decisions | Proposal → review → implement cycle for architecture changes |

## Skills (Auto-Activated)

[Agent Skills](https://agentskills.io) are an open standard for giving AI agents specialized knowledge. Skills are **auto-discovered** — the agent reads each skill's description at startup and automatically activates the relevant skill when your task matches.

**Supported by**: Claude Code, Cursor, Gemini CLI, VS Code, and [many more](https://agentskills.io/home).

### Available Skills

| Skill | Activates when you... |
|-------|----------------------|
| `pace-cluster` | Ask about PACE, Slurm, sbatch, GPUs, job scripts, cluster setup |
| `experiment` | Want to create, run, validate, or document experiments |
| `research-docs` | Want to capture concepts, papers, tools, or research notes |

### How Skills Work

1. **Discovery**: At startup, the agent reads the `name` and `description` from each `skills/*/SKILL.md`
2. **Activation**: When your task matches a skill's description, the agent loads the full instructions
3. **Execution**: The agent follows the instructions, referencing additional files as needed

You don't need to do anything special — just describe what you want and the agent picks up the right skill.

### Example

```
You: "I need to request a GPU on PACE for training"
→ Agent auto-loads pace-cluster skill
→ Agent knows about QoS policies, GPU types, salloc flags, venv strategies
→ Agent writes the correct salloc command for you
```

### Creating New Skills

Create a directory under `skills/` with a `SKILL.md` file:

```
skills/my-skill/
├── SKILL.md          # Required: metadata + instructions
├── scripts/          # Optional: executable code
├── references/       # Optional: detailed docs
└── assets/           # Optional: templates, data
```

The `SKILL.md` must have YAML frontmatter with `name` and `description`:

```yaml
---
name: my-skill
description: When to activate this skill and what it does.
---

# Instructions for the agent
...
```

See the [Agent Skills specification](https://agentskills.io/specification) for full details.

## Slash Commands (User-Invoked)

Slash commands are **Claude Code-specific** shortcuts. The user explicitly types a command and Claude follows the defined steps. These are useful when you know exactly what workflow you want.

### Available Commands

| Command | Purpose | Output |
|---------|---------|--------|
| `/docs:concept` | Create a research concept doc | `docs/concepts/<name>.md` |
| `/docs:reference` | Capture paper or research output | `docs/references/<type>/<name>.md` |
| `/docs:vendor` | Document an external tool | `docs/vendor/<name>.md` |
| `/experiment:proposal` | Scaffold a new experiment | `user/<name>/experiments/NNN-name/` |
| `/experiment:result` | Record experiment results | `user/<name>/experiments/NNN-name/results.md` |
| `/experiment:validate` | Check experiment completeness | Validation report |
| `/opsx:new` | Create a new change proposal | `openspec/changes/<name>/` |
| `/opsx:ff` | Fast-forward all planning artifacts | All planning docs |
| `/opsx:apply` | Implement approved tasks | Code changes |
| `/opsx:archive` | Archive a completed change | `openspec/changes/archive/` |

### How They Work

Slash commands are defined as `.md` files in `.claude/commands/`. Each file contains step-by-step instructions that Claude follows when the command is invoked.

### Creating Custom Commands

Add a `.md` file to `.claude/commands/<category>/`:

```markdown
---
name: My Command
description: What this command does.
category: Custom
tags: [custom]
---

**Purpose**
[What this command accomplishes]

**Steps**
1. [Step 1]
2. [Step 2]
```

The command becomes available as `/<category>:<filename>`.

## OpenSpec (Change Management)

OpenSpec uses the **OPSX workflow** — a fluid, iterative approach to spec-driven development. It replaces the legacy phase-locked system with actions that can occur in any sequence. Use it to ensure big decisions (architecture changes, new features, breaking changes) are proposed and reviewed before implementation.

### Setup

After forking, run `openspec init` to generate OPSX commands in `.claude/commands/openspec/`. Configuration lives in `openspec/config.yaml`.

### When to Use OpenSpec

- Architecture changes
- Breaking changes
- New major capabilities
- Big performance or security work

### How It Works

```
/opsx:explore → /opsx:new → /opsx:continue or /opsx:ff → /opsx:apply → /opsx:archive
```

1. **Explore**: Investigate ideas before committing (`/opsx:explore`)
2. **Create**: Initialize a change with proposal, specs, design, tasks (`/opsx:new` + `/opsx:ff`)
3. **Implement**: Work through tasks (`/opsx:apply`)
4. **Verify**: Validate implementation quality (`/opsx:verify`)
5. **Archive**: Finalize and close out (`/opsx:archive`)

See [OpenSpec Guide](openspec.md) for full details.

## Skills vs Slash Commands

Skills and slash commands overlap in scope but differ in UX:

| | Skills | Slash Commands |
|---|---|---|
| **Trigger** | Automatic (task matching) | Manual (user types `/command`) |
| **Platform** | Any compatible agent | Claude Code only |
| **Best for** | "Help me with X" (agent figures out how) | "Do exactly this workflow" |
| **Format** | Open standard (`SKILL.md`) | Claude Code-specific (`.claude/commands/`) |

**In practice**: Skills handle most cases automatically. Use slash commands when you want to explicitly trigger a specific workflow step (e.g., `/experiment:validate` to run the checklist).
