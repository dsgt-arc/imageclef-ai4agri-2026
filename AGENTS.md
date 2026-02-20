# OpenSpec (OPSX Workflow)

This project uses [OpenSpec](https://github.com/Fission-AI/OpenSpec) for spec-driven change management. The OPSX workflow replaces the legacy phase-locked approach with fluid, iterative actions.

When a request involves architecture changes, new capabilities, breaking changes, or big decisions:
1. Use `/opsx:explore` to investigate before committing
2. Use `/opsx:new` or `/opsx:ff` to create a change proposal
3. Use `/opsx:apply` to implement approved changes
4. Use `/opsx:archive` to finalize completed work

Configuration: `openspec/config.yaml`
Changes: `openspec/changes/<name>/`

See [OpenSpec Guide](docs/guides/openspec.md) for the full workflow.

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

# PACE Cluster (Phoenix)

When running on the Georgia Tech PACE Phoenix cluster:

1. **Login vs compute nodes**: Login nodes are only for requesting resources via `salloc`. Do all work on compute nodes.
2. **Scratch disk**: Use `~/scratch` for data, checkpoints, and caches (home directory has limited quota)
3. **Cache redirect**: Set `XDG_CACHE_HOME="$HOME/scratch/.cache"` in `~/.bashrc` to redirect all tool caches to scratch
4. **venv location**: Prefer `$TMPDIR/.venv` on compute nodes for fast local SSD access (must reinstall each allocation). Alternatively use `~/scratch/.venv` for persistence.
5. **Environment variable**: Set `UV_PROJECT_ENVIRONMENT` to your chosen `.venv` path before running `uv sync`
6. **Job scripts**: Use `uv run` inside sbatch scripts for reproducible environments
