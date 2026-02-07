# Getting Started

How to fork this template, set up your environment, and run your first experiment.

## 1. Fork the Template

1. Fork this repository on GitHub
2. Rename placeholders throughout:
   - `ruff.toml`: Update `known-first-party` to match your package name
   - `openspec/project.md`: Fill in your project context
   - `README.md`: Update project title and description

## 2. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that replaces pip.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv
```

## 3. Create Your User Workspace

Each team member has their own workspace with its own `pyproject.toml` for dependencies:

```bash
# Copy the example workspace
cp -r user/example user/<your-name>
```

Edit `user/<your-name>/pyproject.toml`:
- Update `name` to `user-<your-name>`
- Update `authors` with your info
- Add the dependencies your experiments need

## 4. Set Up Your Environment

```bash
# Create a virtual environment
uv venv .venv

# Activate it
source .venv/bin/activate

# Install your workspace dependencies
uv sync --package user-<your-name>
```

## 5. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run automatically on every commit.

## 6. Run Your First Experiment

Use Claude Code to create an experiment proposal:

```
/experiment:proposal
```

This will guide you through:
- Choosing an experiment name and ID
- Writing a hypothesis
- Defining method and success criteria
- Creating a task checklist

## Next Steps

- Read [Claude Commands](claude-commands.md) to learn about all available slash commands
- Read [OpenSpec](openspec.md) to understand spec-driven development
- If using PACE cluster, read [PACE Setup](pace-setup.md)
