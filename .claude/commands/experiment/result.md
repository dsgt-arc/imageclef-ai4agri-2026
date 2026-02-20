---
name: Experiment Result
description: Record observations and analysis for a completed experiment.
category: Research
tags: [experiment, result]
---

**Purpose**

Add results to a completed experiment. This documents what happened, whether the hypothesis was confirmed or rejected, and what comes next.

**Guardrails**

- Only add results to experiments that have been run.
- Be honest about outcomes; negative results are valuable.
- Include both quantitative metrics and qualitative observations.
- Document confounders and issues that may have affected results.
- Always specify next steps.

**Steps**

1. Identify the experiment to document:
   - Experiments are in `user/<name>/experiments/`
   - List experiments: `ls user/<name>/experiments/`
   - Verify the experiment has a `proposal.md`
   - Check that results haven't already been recorded

2. Create `results.md` in the experiment directory using `docs/_templates/experiment-result.md`:
   - Add Date Completed
   - Write a one-line Summary answering: what happened?
   - Set Verdict: `confirmed`, `rejected`, or `inconclusive`
   - Record Observations:
     - Quantitative: metrics table with baseline, result, delta
     - Qualitative: bullet points of notable observations
   - Write Analysis explaining what the results mean
   - Note Confounders that may have affected outcomes
   - List Artifacts with locations (logs, checkpoints, code)
   - Define Next Steps as actionable items

3. Mark all tasks in `tasks.md` as complete.

4. Update the proposal's Status to `completed`.

5. Run `/experiment:validate` to verify everything is in order.

6. If the experiment leads to code changes, consider creating an OpenSpec proposal.

7. If the experiment leads to new hypotheses, consider creating new concepts or experiments.

**Template Location**

`docs/_templates/experiment-result.md`

**Verdicts**

| Verdict | Meaning |
|---------|---------|
| `confirmed` | Hypothesis was supported by the data |
| `rejected` | Hypothesis was refuted by the data |
| `inconclusive` | Results were ambiguous or confounded |

**Example**

```bash
# Adding results to an experiment
user/<name>/experiments/001-encoder-benchmark/results.md
```

**Linking Forward**

Results should always lead somewhere:

```
results.md
    ↓
├── New concept (unexpected finding)
├── New experiment (follow-up question)
├── OpenSpec change (implement what worked)
└── Archive concept (idea fully explored)
```
