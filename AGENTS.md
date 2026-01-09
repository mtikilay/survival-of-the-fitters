# AGENTS.md - Implementation Tasks

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
# AGENTS.md — Vocational Trades Training Portfolio Research

## Goal
Build a research pipeline that constructs and evaluates an equity portfolio with strong exposure to **hands-on vocational training** (trades, transport tech, allied health, aged care, community services), while explicitly excluding:
- IT / software / coding education
- Traditional higher education degree businesses

Core holdings must be near pure-plays (>=50% revenue from relevant vocational training). Satellites may be up to 50% of portfolio but must still be strongly aligned to hands-on workforce training or credentialing.

## Current Core Universe
Universe is in `data/universe.csv` with tickers and `bucket` = core/satellite.
Initial core:
- LINC, UTI, ATGE, NXD.AX, SIT.AX, EDU.AX

## Required Improvements (Top Priority)
1. **Fix returns calculation**
   - `to_returns()` in `src/vctp/fetch.py` is messy; refactor to a clean, correct implementation.
   - Support both log and simple returns robustly.

2. **Add data quality / survivorship safeguards**
   - Warn on tickers with too many missing values.
   - Align price series and drop assets with insufficient history (configurable threshold).
   - Add FX handling option (base currency GBP) if feasible, but optional.

3. **Add robust optimization**
   Implement:
   - Equal-weight
   - Risk parity (or inverse-vol + correlation-aware approximation)
   - Minimum variance
   - Maximum diversification ratio
   Each must support constraints:
   - long-only
   - max weight per name
   - satellite sleeve cap: sum(weights where bucket==satellite) <= 0.50
   - optional: min core weight >= 0.50

4. **Add reporting outputs**
   - Write `outputs/weights_<strategy>.csv`
   - Write `outputs/summary.csv` with annualized return/vol, max drawdown, etc.
   - Add a simple HTML report (optional) summarizing:
     - Universe stats
     - Correlation heatmap (matplotlib)
     - Portfolio weights and backtest curve

5. **Backtest**
   Implement a simple monthly-rebalance backtest:
   - input: weights function and constraints
   - output: equity curve, drawdowns, turnover, CAGR/vol
   Ensure it works with sparse tickers.

## Satellite Expansion Workflow (Second Priority)
Add a script `scripts/screen_satellites.py` that helps expand the satellite list without violating constraints:
- Start from a hand-curated candidate list (manual input) and compute:
  - correlation to core
  - incremental diversification benefit
  - incremental drawdown reduction
- Enforce: no IT/coding training exposure; no degree-centric higher ed.
- Output: ranked candidates by diversification + stability.

Do NOT auto-scrape the web. Prefer a manual list in `data/satellite_candidates.csv`.

## Style & Safety Constraints
- Keep code runnable with minimal dependencies.
- Prefer pandas/numpy/scipy/matplotlib; no heavy quant libs.
- Add type hints and docstrings.
- Add unit tests if time permits (pytest optional).
- No web browsing or scraping in code.
- Never include or recommend coding bootcamp/IT training companies.

## Definition: "Hands-on vocational training"
Examples include:
- Automotive, diesel, collision repair
- HVAC, welding, electrical, plumbing
- Construction trades
- Aged care, allied health assistants, practical nursing (non-degree)
- Industrial compliance / safety certifications for manual roles
Exclude:
- Software engineering bootcamps
- IT certs as primary business
- University degree programs / general higher education

## Deliverables
- Refactor and correct the pipeline.
- Add optimizers and backtest.
- Produce CSV outputs and (optional) HTML summary.
- Keep it easy to iterate by editing CSV universe files.
