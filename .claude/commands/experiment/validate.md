---
name: Experiment Validate
description: Validate an experiment's structure and completeness.
category: Research
tags: [experiment, validate]
---

**Purpose**

Validate that an experiment follows the required structure and has all necessary components. Run before requesting approval or marking complete.

**Usage**

```
/experiment:validate [experiment-id]
```

If no experiment-id provided, list available experiments for selection.

**Validation Checks**

1. **Structure** (required files):
   - [ ] `pyproject.toml` exists
   - [ ] `proposal.md` exists
   - [ ] `tasks.md` exists
   - [ ] `artifacts/` directory exists

2. **Proposal completeness**:
   - [ ] Has `Status:` field with valid value (proposed|approved|in-progress|completed|abandoned)
   - [ ] Has `## Hypothesis` section (non-empty)
   - [ ] Has `## Method` section with Approach, Setup, Procedure
   - [ ] Has `## Evaluation` section with Metrics and Success Criteria

3. **Tasks format**:
   - [ ] Has at least one task group (## heading)
   - [ ] Tasks use checkbox format (`- [ ]` or `- [x]`)
   - [ ] Tasks are numbered (e.g., `1.1`, `2.3`)

4. **Status-specific checks**:
   - If `in-progress`: At least one task should be checked
   - If `completed`: All tasks should be checked, `results.md` should exist

**Steps**

1. If experiment-id not provided:
   - Experiments are in `user/<name>/experiments/`
   - List experiments: `ls user/<name>/experiments/`
   - Ask user to specify which to validate

2. Check structure:
   ```bash
   ls user/<name>/experiments/{id}/
   ```

3. Read and validate `proposal.md`:
   - Check Status field
   - Verify required sections exist and are non-empty

4. Read and validate `tasks.md`:
   - Count total tasks and completed tasks
   - Verify format

5. Apply status-specific checks

6. Report results:
   ```
   Experiment: NNN-name
   Status: [status]

   ✓ proposal.md exists
   ✓ tasks.md exists
   ✓ Hypothesis defined
   ✗ Missing Success Criteria

   Tasks: 3/10 completed

   Issues found: 1
   ```

**Example Output**

```
Validating: 001-encoder-benchmark

Structure:
  ✓ pyproject.toml
  ✓ proposal.md
  ✓ tasks.md
  ✓ artifacts/

Proposal:
  ✓ Status: proposed
  ✓ Hypothesis
  ✓ Method (Approach, Setup, Procedure)
  ✓ Evaluation (Metrics, Success Criteria)

Tasks:
  ✓ Format valid
  ○ 0/12 completed (expected for 'proposed' status)

Result: VALID
```
