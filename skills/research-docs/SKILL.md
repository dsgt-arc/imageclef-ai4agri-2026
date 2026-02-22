---
name: research-docs
description: Use this skill when the user wants to document research knowledge. This includes capturing concepts or hypotheses, summarizing papers or literature, recording AI deep-research outputs, or documenting external tools, datasets, or libraries. Activate when the user mentions papers, ideas, research notes, vendor tools, or asks to write something down for the team.
metadata:
  author: dsgt-arc
  version: "1.0"
---

# Research Documentation

## Overview

The project uses a structured documentation workflow to capture and organize research knowledge. Documents flow through a pipeline:

```
Reference → Concept → Experiment Proposal → Results
    ↓          ↓              ↓                ↓
 literature  ideas      test hypothesis    document
```

## Document Types

### Concepts (`docs/concepts/`)

Research ideas or hypotheses worth exploring.

**When to create**: The user has a testable idea, an observation, or a technique they want to explore.

**Template**: [docs/_templates/concept.md](../../docs/_templates/concept.md)

**Naming**: kebab-case (e.g., `audio-mixup-augmentation.md`)

**Required sections**:
- Summary of the idea
- Motivation / why it might work
- Related work or references
- How it could be tested

### References

#### Literature (`docs/references/literature/`)

Papers, articles, and external publications.

**When to create**: The user reads a paper or article relevant to the project.

**Template**: [docs/_templates/reference.md](../../docs/_templates/reference.md)

**Naming**: descriptive kebab-case (e.g., `ast-audio-spectrogram-transformer.md`)

**Required sections**:
- Citation and link
- Key findings
- Relevance to this project
- Notable methods or techniques

#### Deep Research (`docs/references/deep-research/`)

AI-generated research syntheses (e.g., from Claude deep research, Perplexity, etc.).

**When to create**: The user runs an AI research query and wants to preserve the output.

**Template**: [docs/_templates/reference.md](../../docs/_templates/reference.md)

**Naming**: `YYYY-MM-DD-descriptive-name.md`

### Vendor Documentation (`docs/vendor/`)

External tools, datasets, libraries, or infrastructure used by the project.

**When to create**: The team adopts a new tool, dataset, or library.

**Template**: [docs/vendor/_TEMPLATE.md](../../docs/vendor/_TEMPLATE.md)

**Naming**: kebab-case (e.g., `pace-phoenix.md`)

**Required sections** (from template frontmatter):
- `name`, `source`, `category`, `relevance`, `status`
- Overview, key components, relevance to pipeline, usage

**After creating**: Add an entry to `docs/vendor/README.md` index table.

## Creating Documents

### Step-by-step

1. Ask the user what type of document (concept, reference, vendor)
2. Ask for the key information
3. Copy the appropriate template
4. Fill in the content
5. Place in the correct directory with proper naming
6. For vendor docs, update `docs/vendor/README.md`

### File Locations

```
docs/
├── concepts/               # Research ideas
│   └── *.md
├── references/
│   ├── literature/         # Papers and articles
│   │   └── *.md
│   └── deep-research/      # AI research outputs
│       └── YYYY-MM-DD-*.md
└── vendor/                 # External tool docs
    ├── README.md           # Index (update when adding)
    ├── _TEMPLATE.md        # Template
    └── *.md
```

## Guidelines

- When the user learns something new, capture it immediately
- Keep documents focused — one idea per concept, one tool per vendor doc
- Always link related documents (e.g., concept references a paper, experiment references a concept)
- Use the templates — they ensure consistency across the team
