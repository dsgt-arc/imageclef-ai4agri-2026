# Documentation

Shared context and documentation for this CLEF project.

---

## Quick Start

**New here?** Start with:
1. [concepts/](concepts/) - Research ideas we're exploring
2. [references/](references/) - Key papers and literature
3. [vendor/](vendor/) - External tools and datasets

---

## Directory Structure

```
docs/
├── README.md           # This file
├── concepts/           # Research ideas and hypotheses
│   └── *.md            # Individual concepts (use /docs:concept)
├── references/
│   ├── literature/     # Papers and articles
│   └── deep-research/  # AI research outputs (use /docs:reference)
├── vendor/             # External tool documentation (use /docs:vendor)
│   └── README.md       # Index of vendored repos
├── guides/             # How-to documentation
│   ├── getting-started.md
│   ├── agent-tooling.md
│   ├── claude-commands.md
│   ├── openspec.md
│   └── pace-setup.md
└── _templates/         # Document templates (don't modify)
    ├── concept.md
    ├── reference.md
    ├── experiment-proposal.md
    ├── experiment-result.md
    ├── experiment-tasks.md
    └── experiment-design.md
```

---

## Concepts (Research Ideas)

| Concept | Description |
|---------|-------------|
| *(use `/docs:concept` to add new concepts)* | |

---

## References

### Literature
- [references/literature/](references/literature/) - Papers and articles

### Deep Research (AI Syntheses)
- [references/deep-research/](references/deep-research/) - AI research outputs

### Vendor Documentation
See [vendor/README.md](vendor/README.md) for external tools and datasets.

---

## Templates

All templates are in [`_templates/`](_templates/). Use the corresponding slash commands to create new documents:

| Template | Slash Command |
|----------|---------------|
| `concept.md` | `/docs:concept` |
| `reference.md` | `/docs:reference` |
| `experiment-proposal.md` | `/experiment:proposal` |
| `experiment-result.md` | `/experiment:result` |
| `experiment-tasks.md` | `/experiment:proposal` (auto-created) |
| `experiment-design.md` | `/experiment:proposal` (optional) |

---

## Guides

| Guide | Description |
|-------|-------------|
| [Getting Started](guides/getting-started.md) | Fork, setup, first experiment |
| [Agent Tooling](guides/agent-tooling.md) | Skills, slash commands, and OpenSpec explained |
| [Claude Commands](guides/claude-commands.md) | Slash command reference |
| [OpenSpec](guides/openspec.md) | OPSX spec-driven development workflow |
| [PACE Setup](guides/pace-setup.md) | PACE cluster environment setup |
