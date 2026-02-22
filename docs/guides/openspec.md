# OpenSpec Workflow (OPSX)

OpenSpec uses the OPSX workflow — a fluid, iterative approach to spec-driven change management. Changes are proposed, reviewed, implemented, and archived through structured artifacts.

## Setup

After forking the template, initialize OpenSpec:

```bash
npm install -g @fission-ai/openspec@latest
openspec init
```

This generates OPSX commands in `.claude/commands/openspec/` and creates project configuration at `openspec/config.yaml`.

## When to Use OpenSpec

Use OpenSpec when you need to:
- Add new features or functionality
- Make breaking changes (API, schema, architecture)
- Introduce new patterns or dependencies
- Make changes that benefit from review before implementation

**Skip OpenSpec for:**
- Bug fixes that restore intended behavior
- Typos, formatting, comments
- Non-breaking dependency updates
- Configuration changes
- Tests for existing behavior

## Decision Tree

```
New request?
├─ Bug fix restoring spec behavior? → Fix directly
├─ Typo/format/comment? → Fix directly
├─ New feature/capability? → Create change
├─ Breaking change? → Create change
├─ Architecture change? → Create change
└─ Unclear? → /opsx:explore first
```

## OPSX Commands

| Command | Purpose |
|---------|---------|
| `/opsx:explore` | Investigate ideas, clarify requirements |
| `/opsx:new` | Initialize a new change |
| `/opsx:continue` | Build next artifact incrementally |
| `/opsx:ff` | Create all planning artifacts at once |
| `/opsx:apply` | Implement approved tasks |
| `/opsx:verify` | Validate implementation quality |
| `/opsx:sync` | Integrate delta specs into main |
| `/opsx:archive` | Finalize and close out a change |

## Workflow

### 1. Explore (optional)

Use `/opsx:explore` as a thinking partner for unstructured investigation. Move to `/opsx:new` once direction is clear.

### 2. Create a Change

```bash
/opsx:new add-data-augmentation
```

This creates `openspec/changes/add-data-augmentation/` with metadata. Then build planning artifacts:

- **Incremental**: `/opsx:continue` — creates one artifact at a time, with review between each
- **Fast-forward**: `/opsx:ff` — creates all planning artifacts (proposal, specs, design, tasks) at once

Use `/opsx:continue` while exploring, `/opsx:ff` when direction is clear.

### 3. Implement

```bash
/opsx:apply add-data-augmentation
```

Works through `tasks.md` sequentially, writing code and checking off items. Can resume if interrupted.

### 4. Verify (optional)

```bash
/opsx:verify add-data-augmentation
```

Checks completeness, correctness, and coherence. Issues are categorized as CRITICAL, WARNING, or SUGGESTION.

### 5. Archive

```bash
/opsx:archive add-data-augmentation
```

Moves the completed change to `openspec/changes/archive/YYYY-MM-DD-add-data-augmentation/` and optionally syncs delta specs.

## Artifact Dependency Model

Artifacts form a dependency graph (DAG). The default `spec-driven` schema:

```
proposal (root)
├── specs (requires: proposal)
├── design (requires: proposal)
└── tasks (requires: specs + design)
```

Artifact states:
- **BLOCKED**: Missing required dependencies
- **READY**: All prerequisites satisfied
- **DONE**: File exists on filesystem

## Project Configuration

`openspec/config.yaml` establishes defaults and project context:

```yaml
schema: spec-driven

context: |
  Tech stack: Python, PyTorch, uv
  Compute: Georgia Tech PACE Phoenix cluster (Slurm)
  Experiments: Per-user workspaces under user/<name>/experiments/
  Testing: pytest

rules:
  proposal:
    - Include compute requirements estimate
  specs:
    - Use Given/When/Then format for scenarios
```

Context gets injected into all artifacts. Schema can be overridden per-change.

## CLI Commands

```bash
openspec init                       # Set up OPSX in project
openspec list                       # List active changes
openspec list --specs               # List specifications
openspec status --change <name>     # Check change progress
openspec schemas                    # List available schemas
openspec archive <name> --yes       # Archive from CLI
```

## Custom Schemas

Create custom artifact workflows:

```bash
openspec schema init my-workflow        # Create interactively
openspec schema fork spec-driven mine   # Fork from existing
openspec schema validate mine           # Verify structure
```

Schemas live in `openspec/schemas/` (project) or `~/.local/share/openspec/schemas/` (user global).

## Full Reference

See the [OpenSpec documentation](https://github.com/Fission-AI/OpenSpec) for complete details.
