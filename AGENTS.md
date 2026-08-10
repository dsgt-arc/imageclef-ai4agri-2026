# Repository notes

This repository contains the DS@GT ARC implementation for the ImageCLEF AI4Agri 2026 Subtask 1 submission.

## Working conventions

- Keep experiment code and notebooks under the relevant directory in [user](user).
- Use descriptive names for experiments, notebooks, and supporting scripts.
- Prefer reproducible configuration and training scripts alongside the experiment code.
- Document major modeling decisions and results in the experiment directory.

## Workflow

```
Reference → Concept → Experiment Proposal → Results
    ↓          ↓              ↓                ↓
 literature  ideas      test hypothesis    document
```

When you learn something new:
1. Capture in `references/` if external knowledge
2. Add to `concepts/` if it's a testable idea
3. Create experiment when ready to test (`/experiment:proposal`)
4. Validate before running (`/experiment:validate`)
5. Document results (`/experiment:result`)

Experiments are tracked per-user in `user/<name>/experiments/`.
