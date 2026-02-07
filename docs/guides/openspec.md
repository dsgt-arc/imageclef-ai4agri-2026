# OpenSpec Workflow

OpenSpec is a spec-driven development approach where changes to the project are proposed, reviewed, implemented, and archived through structured documents.

## When to Use OpenSpec

Use OpenSpec when you need to:
- Add new features or functionality
- Make breaking changes (API, schema, architecture)
- Introduce new patterns or dependencies
- Make changes that benefit from review before implementation

**Skip OpenSpec for:**
- Bug fixes that restore intended behavior
- Typos, formatting, comments
- Non-breaking dependency updates
- Configuration changes
- Tests for existing behavior

## Decision Tree

```
New request?
├─ Bug fix restoring spec behavior? → Fix directly
├─ Typo/format/comment? → Fix directly
├─ New feature/capability? → Create proposal
├─ Breaking change? → Create proposal
├─ Architecture change? → Create proposal
└─ Unclear? → Create proposal (safer)
```

## Three-Stage Workflow

### Stage 1: Create Proposal (`/openspec:proposal`)

1. Review current state: `openspec list`, `openspec list --specs`
2. Choose a verb-led change ID (e.g., `add-data-augmentation`)
3. Scaffold under `openspec/changes/<id>/`:
   - `proposal.md` - Why and what changes
   - `tasks.md` - Implementation checklist
   - `design.md` - Technical decisions (optional)
   - `specs/<capability>/spec.md` - Requirement deltas
4. Validate: `openspec validate <id> --strict`
5. Request review and approval

### Stage 2: Implement Change (`/openspec:apply`)

1. Read proposal, design, and tasks
2. Work through tasks sequentially
3. Keep changes minimal and focused
4. Update task checklist as you go
5. Mark all tasks complete when done

### Stage 3: Archive Change (`/openspec:archive`)

1. Confirm the change ID
2. Run `openspec archive <id> --yes`
3. Verify specs updated and change archived
4. Validate: `openspec validate --strict`

## Spec Format Quick Reference

```markdown
## ADDED Requirements
### Requirement: Feature Name
The system SHALL provide [capability].

#### Scenario: Success case
- **WHEN** user performs action
- **THEN** expected result

## MODIFIED Requirements
### Requirement: Existing Feature
[Complete updated requirement text]

## REMOVED Requirements
### Requirement: Old Feature
**Reason**: [Why removing]
**Migration**: [How to handle]
```

## CLI Commands Cheat Sheet

```bash
openspec list                    # List active changes
openspec list --specs            # List specifications
openspec show <item>             # View change or spec details
openspec validate <id> --strict  # Validate a change
openspec archive <id> --yes      # Archive after deployment
```

## Full Reference

See [`openspec/AGENTS.md`](../../openspec/AGENTS.md) for the complete OpenSpec workflow documentation.
