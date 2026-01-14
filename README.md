# Survival of the Fitters

A portfolio analysis tool for vocational training companies in the post-AI society.

## Overview

This project analyzes and backtests investment portfolios focused on vocational training companies specializing in:
- Healthcare vocational training (nursing, medical technicians)
- Skilled trades (construction, industrial, HVAC)
- Transportation (automotive, CDL, aviation)
- Allied health careers

**Important:** This portfolio specifically focuses on hands-on vocational skills, not IT/coding training, which are actively shunned.

### 2025 performance

<img width="901" height="1600" alt="image" src="https://github.com/user-attachments/assets/56e63dfb-8339-404a-a655-7d04608d57bc" />


## Installation

### Option 1: Virtual Environment (Recommended)

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Editable Install

```bash
# Install as an editable package
pip install -e .
```

Requirements:
- Python >= 3.10
- pandas >= 2.2
- numpy >= 2.0
- yfinance >= 0.2.50
- matplotlib >= 3.8
- scipy >= 1.12
- pyyaml >= 6.0.1

## Usage

### Run Full Pipeline

```bash
python scripts/run_pipeline.py \
  --start 2020-01-01 \
  --end 2023-12-31 \
  --strategy equal_weight \
  --rebalance Q \
  --initial-capital 100000 \
  --output output
```

**Arguments:**
- `--universe`: Path to universe CSV (default: `data/universe.csv`)
- `--start`: Start date YYYY-MM-DD (default: 3 years ago)
- `--end`: End date YYYY-MM-DD (default: today)
- `--strategy`: Portfolio strategy (`equal_weight` or `max_div`)
- `--rebalance`: Rebalancing frequency (`M`=monthly, `Q`=quarterly)
- `--initial-capital`: Initial portfolio value (default: 100000)
- `--output`: Output directory (default: `output`)

### Test with Synthetic Data

If you don't have internet access or want to test the pipeline:

```bash
python scripts/test_pipeline.py
```

This generates synthetic price data and runs both strategies.

## Strategies

### Equal Weight
Allocates equal weight to all assets in the universe. Simple, diversified approach.

### Max Diversification
Optimizes portfolio weights to maximize the diversification ratio:
- Ratio = (sum of weighted volatilities) / (portfolio volatility)
- Subject to weight constraints (5-30% per asset)
- Rebalanced periodically

## Outputs

The pipeline generates the following outputs:

### CSV Files
- `backtest_results.csv`: Daily portfolio values
- `weights_over_time.csv`: Portfolio weights at each rebalance date

### Visualizations
- `equity_curve.png`: Portfolio value over time
- `allocation.png`: Current portfolio allocation pie chart
- `correlation_heatmap.png`: Asset return correlations

### Reports
- `summary.txt`: Performance metrics summary including:
  - Total and annualized returns
  - Volatility and Sharpe ratio
  - Max drawdown and Calmar ratio
  - Portfolio weights

## Project Structure

```
survival-of-the-fitters/
├── data/
│   └── universe.csv          # Universe of training company tickers
├── src/sotf/
│   ├── universe.py           # Load and validate universe
│   ├── fetch.py              # Fetch price data, calculate returns
│   ├── analytics.py          # Return/vol/correlation metrics
│   ├── optimize.py           # Portfolio optimization strategies
│   ├── backtest.py           # Backtesting engine
│   └── report.py             # Generate reports and charts
├── scripts/
│   ├── run_pipeline.py       # Main execution script
│   └── test_pipeline.py      # Test with synthetic data
├── pyproject.toml            # Project dependencies
└── AGENTS.md                 # Implementation task priorities
```

## Universe

Current universe includes 6 vocational training companies:

| Ticker | Region | Focus |
|--------|--------|-------|
| LINC | US | Trades + allied health career schools |
| UTI | US | Transportation + skilled trades |
| ATGE | US | Healthcare vocational (nursing/medical) |
| NXD.AX | AU | Vocational training (trade/service) |
| SIT.AX | AU | Skilled trades (industrial) |
| EDU.AX | AU | Vocational (aged care/community) |

## License

See LICENSE file for details.
