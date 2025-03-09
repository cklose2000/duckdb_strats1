# DuckDB-Centric Backtesting Framework

A comprehensive backtesting system for financial strategies using DuckDB as the core database engine. This framework emphasizes SQL-based transformations, metadata tracking, and reproducible research.

## System Architecture

The system is organized around a central DuckDB database with metadata tables that track all system objects:

- **scripts** - Stores all Python/R scripts with execution metadata
- **queries** - Stores all SQL queries with purpose and results summary
- **models** - Tracks algorithmic models with parameters and performance
- **datasets** - Catalogs all data sources and transformations
- **executions** - Logs all backtests with parameters and results

## Core Principles

1. Execute ALL data transformations as SQL inside DuckDB
2. Store intermediate results in database tables, not CSV files
3. Auto-log all scripts and queries to metadata tables
4. Use transactions for multi-step operations
5. Query metadata to check existing functionality before building new features

## Repository Structure

```
/
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
    ├── analysis/           # Analysis and reporting utilities
    ├── db_manager.py        # Database connection utilities
    └── metadata_logger.py   # Automatic logging functions
```

## Getting Started

### Prerequisites

- Python 3.7+
- DuckDB
- Required packages: pandas, scikit-learn, yfinance (for sample data)

### Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install duckdb pandas scikit-learn yfinance
```

3. Initialize the database and run migrations:

```bash
python -m utils.initialize_db
```

### Example Workflow

1. **Data Ingestion**: Load stock data into the database

```bash
python -m scripts.data_ingestion.ingest_stock_data
```

2. **Feature Engineering**: Create technical indicators and features

```bash
python -m scripts.feature_engineering.create_technical_indicators
```

3. **Model Training**: Train a return prediction model

```bash
python -m scripts.models.predict_returns
```

4. **Analyze Results**: View backtest results and trade details

```bash
python -m utils.analysis.view_trades
python -m utils.analysis.view_backtest_details
```

5. **View Metadata**: Run metadata queries to understand system state

```bash
duckdb -c "ATTACH 'db/backtest.ddb' AS db; RUN 'queries/view_metadata.sql';"
```

## Key Features

### Metadata-Driven Development

Every object in the system is discoverable through metadata:

- Query `metadata_scripts` to find all available data ingestion scripts
- Query `metadata_datasets` to see transformation lineage
- Query `metadata_models` to review model performance

### SQL-First Approach

All transformations happen in SQL:

- Window functions for time-series analysis
- Complex joins for feature engineering
- Statistical calculations using DuckDB's built-in functions

### Automatic Logging

All activities are automatically logged to metadata tables:

- Script executions with timing information
- Query performance and results summaries
- Model training metrics and parameters

### Analysis and Reporting

The system includes utilities for analyzing backtest results:

- **Trade Summary**: View all trades with entry/exit prices and returns
- **Detailed Analysis**: Analyze the market context around each trade
- **Performance Metrics**: Calculate win rates, returns, and other metrics

To analyze backtest results:

```bash
# View a summary of all trades
python -m utils.analysis.view_trades

# See detailed analysis of trades with surrounding context
python -m utils.analysis.view_backtest_details
```

## Development Workflow

1. **View Existing Components**: Query metadata tables to discover existing functionality
2. **Implement SQL Logic**: Define views and functions in SQL where possible
3. **Create Python Wrappers**: For orchestration and visualization only
4. **Log Everything**: Ensure all new components log themselves to metadata

## License

This project is licensed under the MIT License - see the LICENSE file for details. 