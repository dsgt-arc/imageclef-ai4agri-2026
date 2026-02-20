---
name: Docs: Reference
description: Capture external knowledge from papers, articles, or deep research.
category: Research Docs
tags: [docs, reference, literature]
---

**Purpose**

Create a reference document to capture external knowledge (papers, articles, deep research outputs) that may inform concepts or experiments.

**Guardrails**

- Focus on extracting actionable insights, not just summarizing.
- Link to existing concepts if the reference informs them.
- Use the correct subdirectory: `deep-research/` for AI research outputs, `literature/` for papers and articles.

**Steps**

1. Determine the reference type:
   - **deep-research**: Outputs from AI deep research tools
   - **paper**: Academic papers (ArXiv, conferences, journals)
   - **article**: Blog posts, documentation, tutorials
   - **documentation**: Official docs, API references

2. Choose a filename using the convention: `YYYY-MM-DD-descriptive-name.md`
   - Use today's date for when the reference was captured
   - Use kebab-case for the descriptive name

3. Create the file in the appropriate directory:
   - `docs/references/deep-research/` for AI research outputs
   - `docs/references/literature/` for papers/articles

4. Use the template from `docs/_templates/reference.md`:
   - Fill in Source (URL or citation)
   - Write a 2-3 sentence Summary
   - Extract 3-5 Key Points
   - Note Relevant Sections with quotes or paraphrased content
   - Add Connections to existing concepts or potential experiments

5. If the reference directly informs an existing concept, update that concept's "Based on" section.

**Template Location**

`docs/_templates/reference.md`

**Example**

```bash
# For a paper on audio classification
docs/references/literature/2025-01-15-attention-audio-spectrogram.md

# For deep research on species detection
docs/references/deep-research/2025-01-15-whale-call-detection-methods.md
```
