# CLEF Project Template

Template repository for DS@GT ARC teams working on CLEF competitions. Fork this repo to get started with structured experiment tracking, agent skills, slash commands, OpenSpec spec-driven development, and PACE cluster tooling.

## Quick Start

| Resource | Link |
|----------|------|
| Research concepts | [docs/concepts/](docs/concepts/) |
| Literature references | [docs/references/](docs/references/) |
| Vendor documentation | [docs/vendor/](docs/vendor/) |
| Guides | [docs/guides/](docs/guides/) |

## Guides

- [Getting Started](docs/guides/getting-started.md) - Fork, setup, first experiment
- [Agent Tooling](docs/guides/agent-tooling.md) - Skills, slash commands, and OpenSpec explained
- [Claude Commands](docs/guides/claude-commands.md) - Slash command reference
- [OpenSpec](docs/guides/openspec.md) - Spec-driven development workflow
- [PACE Setup](docs/guides/pace-setup.md) - PACE cluster environment setup

## Repository Structure

```
clef-project-template/
├── skills/                     # Agent Skills (auto-discovered by compatible agents)
│   ├── pace-cluster/           # PACE Phoenix cluster operations
│   ├── experiment/             # Experiment lifecycle management
│   └── research-docs/          # Research documentation workflow
├── .claude/                    # Claude Code config & slash commands
│   ├── settings.json
│   └── commands/
│       ├── docs/               # /docs:concept, /docs:reference, /docs:vendor
│       ├── experiment/         # /experiment:proposal, /experiment:result, /experiment:validate
│       └── openspec/           # OPSX commands (empty until you run `openspec init`)
├── docs/
│   ├── README.md               # Documentation index
│   ├── guides/                 # How-to documentation
│   ├── _templates/             # Document templates (don't modify)
│   ├── concepts/               # Research ideas
│   ├── references/             # Papers and AI research outputs
│   └── vendor/                 # External tool documentation
├── openspec/
│   ├── config.yaml             # Project context and schema (fill in after forking)
│   ├── specs/                  # Built specifications
│   └── changes/                # Change proposals
├── user/
│   └── example/                # Example user workspace
│       └── experiments/
│           └── 000-template/   # Template experiment
│               └── pyproject.toml  # Experiment-level dependencies
├── vendor/                     # Vendored submodules / external repos
├── pyproject.toml              # Workspace root (discovers experiments)
├── AGENTS.md                   # AI assistant guidelines
├── CLAUDE.md                   # Symlink → AGENTS.md
├── ruff.toml                   # Ruff linter/formatter config
├── .pre-commit-config.yaml     # Pre-commit hooks
├── .gitignore
└── LICENSE
```

## Setup

### 1. Fork and Rename

Fork this repository and update the placeholders:

- `ruff.toml`: Change `known-first-party` to your package name
- `openspec/config.yaml`: Fill in your project context and tech stack
- This `README.md`: Update title and description

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create Your User Workspace

```bash
cp -r user/example user/<your-name>
```

Edit the `pyproject.toml` inside your first experiment (e.g., `user/<your-name>/experiments/001-my-experiment/pyproject.toml`) with your name and dependencies, then:

```bash
# Create virtual environment
uv venv .venv

# Activate it
source .venv/bin/activate

# Install experiment dependencies
uv sync --package <experiment-name>
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

## Agent Tooling

This template uses three systems for AI-assisted workflows. See [Agent Tooling Guide](docs/guides/agent-tooling.md) for full details.

### Skills (Auto-Activated)

[Agent Skills](https://agentskills.io) are auto-discovered by compatible agents (Claude Code, Cursor, Gemini CLI, etc.). Just describe what you need and the agent loads the right skill.

| Skill | Activates when you... |
|-------|----------------------|
| `pace-cluster` | Ask about PACE, Slurm, GPUs, job scripts |
| `experiment` | Want to create, run, or document experiments |
| `research-docs` | Want to capture concepts, papers, or tools |

### Slash Commands (User-Invoked)

Claude Code-specific shortcuts for explicit workflow triggers:

| Command | Purpose |
|---------|---------|
| `/docs:concept` | Create a research concept document |
| `/docs:reference` | Capture knowledge from papers/articles |
| `/docs:vendor` | Document an external tool or dataset |
| `/experiment:proposal` | Propose a new experiment |
| `/experiment:result` | Record experiment results |
| `/experiment:validate` | Validate experiment structure |
| `/opsx:explore` | Investigate ideas before committing to a change |
| `/opsx:new` | Create a new change proposal |
| `/opsx:ff` | Fast-forward all planning artifacts |
| `/opsx:apply` | Implement an approved change |
| `/opsx:archive` | Archive a completed change |

### OpenSpec (Change Management)

Spec-driven development using the [OPSX workflow](https://github.com/Fission-AI/OpenSpec). Run `openspec init` after forking to generate skills and commands. See [OpenSpec Guide](docs/guides/openspec.md).

## Contributing

1. Create your user workspace under `user/<your-name>/`
2. Use slash commands to create structured documents
3. Follow the research workflow: Reference → Concept → Experiment → Results
4. See [AGENTS.md](AGENTS.md) for AI assistant conventions
