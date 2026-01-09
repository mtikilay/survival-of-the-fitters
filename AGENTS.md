# AGENTS.md - Implementation Tasks

**Related docs:** [CLAUDE.md](./CLAUDE.md) | [.github/copilot-instructions.md](./.github/copilot-instructions.md)

## Beads Workflow (MANDATORY)

**Every piece of work MUST have a beads issue.** No exceptions.

- `bd ready` - Find work with no blockers
- `bd create --title="..." --type=task|bug|feature --priority=2` - Create new issues
- `bd update <id> --status=in_progress` - Claim work before starting
- `bd close <id>` - Mark complete when done

## Priority Order

### P0: Core Data Infrastructure (Issue #2)
1. Create `data/universe.csv` with vocational training company tickers
2. Set up `pyproject.toml` with minimal dependencies
3. Implement `src/sotf/universe.py` - Load and validate universe data
4. Implement `src/sotf/fetch.py` - Fetch price data and calculate returns

### P1: Analytics & Portfolio Optimization
5. Implement `src/sotf/analytics.py` - Calculate returns, volatility, correlation metrics
6. Implement `src/sotf/optimize.py` - Equal weight and max diversification strategies

### P2: Backtesting Framework
7. Create backtesting engine to simulate portfolio performance over time
8. Support rebalancing logic (monthly/quarterly)
9. Calculate performance metrics (returns, Sharpe ratio, drawdowns)

### P3: Reporting & Visualization
10. Implement `src/sotf/report.py` - Generate performance reports
11. Create CSV outputs for backtest results
12. Add visualization charts (equity curves, allocation pie charts, correlation heatmaps)

### P4: Execution Pipeline
13. Implement `scripts/run_pipeline.py` - Main entry point to run full analysis
14. Add command-line arguments for date ranges and strategy selection

## Constraints
- Keep dependencies minimal (only: pandas, numpy, yfinance, matplotlib, scipy, pyyaml)
- **DO NOT** add any IT/coding/software training company names to universe
- Focus on vocational trades: healthcare, skilled trades, transportation, allied health

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create beads issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
