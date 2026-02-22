---
name: Docs: Concept
description: Explore and document an idea or hypothesis before experimenting.
category: Research Docs
tags: [docs, concept, hypothesis]
---

**Purpose**

Create a concept document to explore an idea, hypothesis, or research direction. Concepts synthesize knowledge from references and lead to testable experiments.

**Guardrails**

- Keep concepts focused on a single idea; split compound ideas into separate documents.
- Ensure at least one testable hypothesis before considering the concept "developing".
- Link back to references that inform the concept.
- Do not conflate concepts with experiments; concepts are exploratory, experiments are tests.

**Steps**

1. Choose a descriptive filename using kebab-case: `descriptive-name.md`
   - No date prefix (concepts evolve over time)
   - Keep names concise but meaningful

2. Create the file at `docs/concepts/<name>.md`

3. Use the template from `docs/_templates/concept.md`:
   - Set Status to `nascent` (new idea, not yet developed)
   - Write a clear 1-2 paragraph Idea description
   - Explain Motivation (why this is worth exploring)
   - Add Background with links to references
   - List Open Questions to investigate
   - Define at least one Testable Hypothesis
   - Fill in Connections (based on, related to, could lead to)

4. Update Status as the concept matures:
   - `nascent` → Initial idea, minimal exploration
   - `developing` → Active exploration, hypotheses forming
   - `mature` → Ready for experiments, clear hypotheses
   - `archived` → Explored and concluded (led to experiments or abandoned)

5. When ready to test, create an experiment proposal that references this concept.

**Template Location**

`docs/_templates/concept.md`

**Status Lifecycle**

```
nascent → developing → mature → archived
   ↓          ↓          ↓
 (abandon) (abandon)  (experiment)
```

**Example**

```bash
# Concept about data augmentation strategies
docs/concepts/audio-mixup-augmentation.md

# Concept about model architecture
docs/concepts/hierarchical-species-classification.md
```
