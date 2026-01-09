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
