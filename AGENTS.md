<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

---

# Documentation Guidelines

## Adding Documentation

Use slash commands to create properly structured documents:

| Command | Purpose | Template |
|---------|---------|----------|
| `/docs:concept` | Research idea or hypothesis | `docs/_templates/concept.md` |
| `/docs:reference` | External knowledge capture | `docs/_templates/reference.md` |
| `/docs:vendor` | External tool/dataset documentation | `docs/vendor/_TEMPLATE.md` |
| `/experiment:proposal` | Experiment proposal | `docs/_templates/experiment-proposal.md` |
| `/experiment:result` | Experiment results | `docs/_templates/experiment-result.md` |
| `/experiment:validate` | Validate experiment structure | - |

## Documentation Structure

```
docs/
├── README.md           # Documentation index
├── concepts/           # Research ideas (use /docs:concept)
├── references/
│   ├── literature/     # Papers and articles
│   └── deep-research/  # AI research outputs (use /docs:reference)
├── vendor/             # External tool docs (use /docs:vendor)
└── _templates/         # Templates (don't modify)

user/<name>/experiments/ # Experiments per user (use /experiment:*)
```

## Conventions

1. **Concepts**: Use kebab-case filenames (e.g., `audio-mixup-augmentation.md`)
2. **References**: Use `YYYY-MM-DD-descriptive-name.md` for deep-research
3. **Vendor**: Use kebab-case filenames (e.g., `tool-name.md`)
4. **Experiments**: Use `NNN-descriptive-name/` directories in `user/<name>/experiments/`

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

---

# PACE Cluster

When running on the Georgia Tech PACE cluster:

1. **Environment variables**: Set `UV_PROJECT_ENVIRONMENT` to a path on scratch disk (not `$HOME`)
2. **Scratch disk**: Use `/storage/ice-shared/dsgt/` or your personal scratch for data and checkpoints
3. **venv location**: Always place `.venv` on scratch, not home directory (quota limited)
4. **Downloads**: Cache HuggingFace models via `HF_HOME` and PyTorch via `TORCH_HOME` on scratch
5. **Job scripts**: Use `uv run` inside sbatch scripts for reproducible environments
