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

# PACE Cluster (Phoenix)

When running on the Georgia Tech PACE Phoenix cluster:

1. **Login vs compute nodes**: Login nodes are only for requesting resources via `salloc`. Do all work on compute nodes.
2. **Scratch disk**: Use `~/scratch` for data, checkpoints, and caches (home directory has limited quota)
3. **Cache redirect**: Set `XDG_CACHE_HOME="$HOME/scratch/.cache"` in `~/.bashrc` to redirect all tool caches to scratch
4. **venv location**: Prefer `$TMPDIR/.venv` on compute nodes for fast local SSD access (must reinstall each allocation). Alternatively use `~/scratch/.venv` for persistence.
5. **Environment variable**: Set `UV_PROJECT_ENVIRONMENT` to your chosen `.venv` path before running `uv sync`
6. **Job scripts**: Use `uv run` inside sbatch scripts for reproducible environments
