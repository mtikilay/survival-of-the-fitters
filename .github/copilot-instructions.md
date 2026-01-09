# Copilot Instructions

**Related docs:** [CLAUDE.md](../CLAUDE.md) | [AGENTS.md](../AGENTS.md)

## Beads Workflow (MANDATORY)

**Every piece of work MUST have a beads issue.** Before starting any task:
- `bd ready` - Find available work
- `bd update <id> --status=in_progress` - Claim the issue
- `bd close <id>` - Mark complete when done

## Project Context

Portfolio analysis tool for vocational training companies. See CLAUDE.md for full architecture and commands.

## Key Constraints

- **NO IT/coding training companies** - Focus on healthcare, skilled trades, transportation, allied health only
- **Minimal dependencies** - pandas, numpy, yfinance, matplotlib, scipy, pyyaml
- **Install via:** `pip install -r requirements.txt`

## Session Completion

Always push changes before ending:
```bash
git pull --rebase
git push
```
