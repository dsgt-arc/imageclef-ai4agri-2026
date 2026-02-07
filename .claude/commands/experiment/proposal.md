---
name: Experiment Proposal
description: Propose and scaffold a new experiment with hypothesis and method.
category: Research
tags: [experiment, proposal]
---

**Purpose**

Create an experiment to test a hypothesis from a concept. Experiments have two phases: proposal (before running) and results (after completion).

**Guardrails**

- Every experiment must have a falsifiable hypothesis.
- Define success/rejection criteria before running the experiment.
- Keep experiments focused; test one thing at a time when possible.
- Link back to the concept that motivated the experiment.

**Steps**

1. Determine the user workspace and next experiment ID:
   - User workspaces are in `user/<name>/experiments/`
   - Ask user for their name if not known
   - List existing experiments: `ls user/<name>/experiments/`
   - Use the next available number: `NNN-descriptive-name/`
   - Example: if 001, 002 exist, use 003

2. Create the experiment directory:
   ```bash
   mkdir -p user/<name>/experiments/NNN-descriptive-name
   mkdir -p user/<name>/experiments/NNN-descriptive-name/artifacts
   ```

3. Create `pyproject.toml` for the experiment:
   - Copy from `user/example/experiments/000-template/pyproject.toml`
   - Update `name` to match the experiment directory name (e.g., `NNN-descriptive-name`)
   - Update `description` and `authors`
   - Add experiment-specific dependencies

4. Create `proposal.md` using `docs/_templates/experiment-proposal.md`:
   - Set Status to `proposed`
   - Write a specific, falsifiable Hypothesis
   - Add Background linking to the motivating concept
   - Define the Method:
     - Approach (high-level description)
     - Setup (data, compute, dependencies, code)
     - Procedure (numbered steps)
     - Variables (independent, dependent, controlled)
   - Specify Evaluation:
     - Metrics with descriptions
     - Baseline for comparison
     - Success Criteria (confirm if / reject if)
   - Note Limitations and assumptions

5. Create `tasks.md` using `docs/_templates/experiment-tasks.md`:
   - Break down the experiment into actionable steps
   - Group by phase: Setup, Implementation, Execution, Analysis
   - Use checkbox format: `- [ ] 1.1 Task description`

6. (Optional) Create `design.md` using `docs/_templates/experiment-design.md` if:
   - Multiple implementation approaches need documentation
   - Architecture decisions should be captured
   - Non-trivial technical setup is required

7. Get approval before running (change Status to `approved`).

8. Update Status to `in-progress` when starting execution.

9. Track progress by checking off tasks in `tasks.md`.

10. After completion, use `/experiment:result` to add results.

11. Use `/experiment:validate` to check completeness before approval.

**Templates**

- `docs/_templates/experiment-proposal.md` - Hypothesis and method
- `docs/_templates/experiment-tasks.md` - Implementation checklist
- `docs/_templates/experiment-design.md` - Technical decisions (optional)
- `docs/_templates/experiment-result.md` - Results documentation

**Status Lifecycle**

```
proposed → approved → in-progress → completed
    ↓         ↓           ↓
 (abandoned) (abandoned) (abandoned)
```

**Directory Structure**

```
user/<name>/experiments/NNN-descriptive-name/
├── pyproject.toml   # Experiment dependencies (required)
├── proposal.md      # Hypothesis and method (required)
├── tasks.md         # Implementation checklist (required)
├── design.md        # Technical decisions (optional)
├── results.md       # Observations (added after completion)
└── artifacts/       # Logs, configs, figures, checkpoints
```

**Example**

```bash
# Experiment on encoder benchmarking
user/<name>/experiments/001-encoder-benchmark/proposal.md

# Experiment on model architecture
user/<name>/experiments/002-ast-vs-cnn-comparison/proposal.md
```
