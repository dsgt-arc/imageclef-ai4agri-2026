# Project Context

## Purpose

<!-- Replace this section with your project's purpose -->
CLEF competition entry for [task name]. This repository contains research code, experiment tracking, and documentation for the DS@GT ARC team.

**Goals:**
1. [Goal 1 - e.g., Develop an effective pipeline for the task]
2. [Goal 2 - e.g., Achieve competitive performance on the evaluation metric]
3. [Goal 3 - e.g., Document findings for future teams]

## Tech Stack

- **Python 3.10+** with `uv` for package management
- **PyTorch / Lightning** for model training (if applicable)
- **numpy / pandas / scikit-learn** for data processing
- **matplotlib** for visualization
- **ruff** for formatting and linting
- **pre-commit** for code quality hooks

## Project Conventions

### Code Style
- Use `ruff` for formatting and linting
- Type hints required for public functions
- Docstrings for modules and classes

### Architecture Patterns
- Per-user workspaces in `user/<name>/`
- Experiments tracked in `user/<name>/experiments/`
- Shared code in project-level packages

### Testing Strategy
- `pytest` for unit tests
- Integration tests on small data subsets

### Git Workflow
- Main branch for stable code
- Feature branches for experiments
- Commit messages: imperative mood, reference tasks

## Domain Context

<!-- Replace this section with your competition/task details -->
- **Task**: [CLEF task name and year]
- **Data**: [Description of input data]
- **Evaluation**: [Evaluation metric]
- **Constraints**: [Any inference/submission constraints]

## Important Constraints

<!-- List any hard constraints your project must respect -->
1. [Constraint 1 - e.g., Inference time limit]
2. [Constraint 2 - e.g., Memory limit]
3. [Constraint 3 - e.g., No internet during inference]

## External Dependencies

<!-- List vendored repos and key external dependencies -->
- [None yet - add as needed]
