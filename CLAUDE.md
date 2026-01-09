# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Related docs:** [AGENTS.md](./AGENTS.md) | [.github/copilot-instructions.md](./.github/copilot-instructions.md)

## Beads Workflow (MANDATORY)

**Every piece of work MUST have a beads issue.** Before starting any task:

1. `bd ready` - Find available work with no blockers
2. `bd show <id>` - Review issue details
3. `bd update <id> --status=in_progress` - Claim the issue
4. Do the work
5. `bd close <id>` - Mark complete when done
6. `git push` - Push code changes (beads auto-synced by daemon)

**Creating new work:** `bd create --title="..." --type=task|bug|feature --priority=2`

## Project Overview

Portfolio analysis and backtesting tool for vocational training companies in the post-AI economy. Analyzes investments in hands-on skills training (healthcare, skilled trades, transportation, allied health) - **NOT IT/coding/software training companies**.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis with real data
python scripts/run_pipeline.py \
  --start 2020-01-01 \
  --end 2023-12-31 \
  --strategy equal_weight \
  --rebalance Q \
  --initial-capital 100000 \
  --output output

# Test with synthetic data (no internet required)
python scripts/test_pipeline.py
```

**CLI Arguments:**
- `--universe`: Path to universe CSV (default: `data/universe.csv`)
- `--start/--end`: Date range (YYYY-MM-DD)
- `--strategy`: `equal_weight` or `max_div`
- `--rebalance`: `M` (monthly) or `Q` (quarterly)
- `--initial-capital`: Starting portfolio value
- `--output`: Output directory (creates timestamped subdirs)

## Architecture

**Data Flow:**
```
Universe CSV → Fetch Prices (yfinance) → Calculate Returns
→ Analytics → Portfolio Optimization → Backtesting → Reports
```

**Core Modules (src/sotf/):**
- `universe.py` - Load/validate ticker universe from CSV
- `fetch.py` - Fetch adjusted close prices via yfinance, convert to returns
- `analytics.py` - Annualized returns, volatility, correlation/covariance matrices
- `optimize.py` - Portfolio strategies: equal_weight (1/N) and max_div (maximize diversification ratio with 5-35% per-asset constraints)
- `backtest.py` - Periodic rebalancing simulation with daily valuations, calculates Sharpe/max drawdown/Calmar ratios
- `report.py` - CSV exports, equity curves, allocation charts, correlation heatmaps

**Universe:** 6 tickers across US (LINC, UTI, ATGE) and Australia (NXD.AX, SIT.AX, EDU.AX)

## Constraints

- Keep dependencies minimal (no new packages without approval)
- Focus on vocational trades only - do not add IT/coding training companies
- Output organized in timestamped directories (YYYYMMDD-HH-MM-SS)
