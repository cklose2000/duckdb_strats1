# DuckDB Automated Trading Strategy Framework

A comprehensive backtesting system using DuckDB for high-performance financial data processing and trading strategy optimization.

## System Architecture

This framework leverages DuckDB's in-process analytics capabilities to efficiently test and optimize trading strategies. The system is designed around these principles:

- All data transformations are executed as SQL inside DuckDB
- Intermediate results are stored in database tables, not CSV files
- All operations are automatically logged to metadata tables
- Transactions are used for multi-step operations
- Metadata is queried to check existing functionality before building new features

### Repository Structure

```
./
├── db/
│   └── backtest.ddb         # Main DuckDB database
├── migrations/              # Schema change scripts
├── queries/                 # SQL query templates
├── scripts/
│   ├── data_ingestion/      # Scripts to load external data
│   ├── feature_engineering/ # Data transformation scripts
│   └── models/              # Algorithmic model implementations
├── notebooks/               # Analysis notebooks
└── utils/
    ├── db_manager.py        # Database connection utilities
    └── metadata_logger.py   # Automatic logging functions
```

## Metadata Schema

The system uses a metadata schema to track all system objects:

- `scripts` table - stores all Python/R scripts with execution metadata
- `queries` table - stores all SQL queries with purpose and results summary
- `models` table - tracks algorithmic models with parameters and performance
- `datasets` table - catalogs all data sources and transformations
- `executions` table - logs all backtests with parameters and results

## Automated Strategy Testing Framework

The automated strategy testing framework allows you to:

1. Generate multiple trading strategies with varied parameters
2. Backtest these strategies against historical market data
3. Analyze performance using key metrics (Sharpe, profit factor, etc.)
4. Optimize strategies using genetic algorithms
5. Store all results in the DuckDB database for future analysis

### Key Components

- **Strategy Generator**: Creates strategy variants by varying parameters like SMA periods, RSI thresholds
- **Parameterized Strategy**: Implementation of trading strategies with configurable parameters
- **Batch Strategy Tester**: Tests multiple strategies in sequence and ranks them by performance
- **Genetic Optimizer**: Evolves and improves strategies using genetic algorithms
- **Automated Backtesting Pipeline**: Orchestrates the entire process

## Usage

### Running the Automated Backtesting Pipeline

```bash
python -m scripts.models.run_automated_backtests --ticker SPY --num 100 --population 50 --generations 10 --plot-top 10
```

Parameters:
- `--ticker`: Ticker symbol to test (default: SPY)
- `--num`: Number of random strategies to test (default: 100)
- `--population`: Genetic algorithm population size (default: 50)
- `--generations`: Number of generations for genetic evolution (default: 10)
- `--plot-top`: Number of top strategies to plot (default: 10)

### Analyzing Results

The results are stored in the DuckDB database and can be analyzed using SQL queries. Sample queries are provided in `queries/analyze_backtest_results.sql`.

## Strategy Parameters

The framework supports the following strategy parameters:

- **SMA Periods**: Fast and slow SMA periods (e.g., 20/50)
- **RSI Period**: Period for calculating RSI (e.g., 14)
- **RSI Thresholds**: Buy/sell thresholds for RSI (e.g., <30 for buy, >70 for sell)
- **Secondary RSI Thresholds**: Secondary thresholds for signal days
- **Volatility Calculation Period**: Period for calculating volatility

## Example Strategy Logic

The multi-timeframe strategy combines signals from different timeframes:

1. Daily timeframe for trend identification (SMA crossovers)
2. Hourly/5min timeframe for entry/exit timing (RSI conditions)
3. Trade action generation based on combined signals

## Database Views

The system creates several views in the database:

- `all_strategies_summary`: Summary of all tested strategies
- `top_strategies`: Top strategies ranked by different metrics
- Various timeframe-specific views for data analysis

## Prerequisites

- Python 3.8+
- DuckDB
- Dependencies in requirements.txt:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - requests
  - python-dotenv

## License

This project is licensed under the MIT License - see the LICENSE file for details. 