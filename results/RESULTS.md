# SOTF Backtest Results: 2020-01-01 to 2026-10-27, $100,000 initial capital

| Metric | Run A: equal_weight, monthly | Run B: max_div, quarterly |
|---|---|---|
| Total return | 322.03% | 766.22% |
| Annualized return | 23.53% | 37.28% |
| Annualized volatility | 34.78% | 32.93% |
| Sharpe ratio | 0.68 | 1.13 |
| Max drawdown | -46.59% | -48.77% |
| Drawdown peak (date, value) | 2026-07-07, $753,723.88 | 2026-07-07, $1,622,216.78 |
| Drawdown trough (date, value) | 2026-10-15, $402,577.23 | 2026-10-15, $831,131.91 |
| Calmar ratio | 0.51 | 0.76 |
| Final portfolio value | $422,031.63 | $866,219.64 |

Note: the console log prints "Peak Date: 2026-10-15" (same as the trough) for both runs. That is a reporting quirk in the script; the peak *values* are correct. The peak dates above were recomputed from `backtest_results.csv`. Source was not modified.

## Final weights

| Ticker | Run A | Run B |
|---|---|---|
| LINC | 25.00% | 29.54% |
| UTI | 25.00% | 14.88% |
| NXD.AX | 25.00% | 28.20% |
| EDU.AX | 25.00% | 27.39% |

## Data provenance

- Universe: 5 tickers (LINC, UTI, ATGE, NXD.AX, EDU.AX).
- Loaded successfully: LINC, UTI, NXD.AX, EDU.AX (4 tickers).
- Failed: ATGE (Yahoo returns "Quote not found" / possibly delisted), so both runs are 4-asset portfolios.
- Price data fetched: 2019-01-14 to 2026-10-26 (2011 days; includes lookback before the 2020-01-01 start). Last available close is 2026-10-26.
- Simulated period: 2020-01-02 to 2026-10-26, 1761 trading days.
- A first attempt of Run A lost LINC to a transient yfinance "database is locked" error; it was discarded and re-run, and the committed Run A has all 4 tickers.

## Interpretation

Max-diversification with quarterly rebalancing did better on both absolute and risk-adjusted terms (Sharpe 1.13 vs 0.68, Calmar 0.76 vs 0.51), with slightly lower volatility and a similar drawdown. The final value is roughly double the equal-weight run's, driven by its higher allocation to LINC (about 30%) and lower UTI weight. These are single-period, in-sample results over a 4-stock universe, so treat the gap with caution.
