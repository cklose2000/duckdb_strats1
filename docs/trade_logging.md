# Trade Logging System

This document outlines the standardized approach to trade logging in the DuckDB-centric backtesting framework.

## Overview

The trade logging system provides a consistent and reliable way to capture, store, and analyze trades across all backtesting operations. It follows the core principles of the DuckDB-centric approach:

- Execute all data transformations as SQL inside DuckDB
- Store trade data in database tables, not CSV files
- Auto-log all trades to metadata tables
- Use transactions for multi-step operations
- Query metadata to understand trade performance

## Trade Data Schema

All trades are stored in the `trade_log` table with the following schema:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `trade_id` | VARCHAR | Unique identifier for the trade (UUID) |
| `strategy_id` | INTEGER | ID of the strategy that generated the trade |
| `date` | TIMESTAMP | Date and time of the trade |
| `symbol` | VARCHAR | Ticker symbol (e.g., 'SPY') |
| `action` | VARCHAR | Trade action (e.g., 'ENTER_LONG', 'EXIT_SHORT') |
| `direction` | VARCHAR | Trade direction ('LONG' or 'SHORT') |
| `price` | DOUBLE | Current price at the time of the trade |
| `size` | DOUBLE | Size of the trade (number of units) |
| `entry_price` | DOUBLE | Price at entry (for exits) |
| `exit_price` | DOUBLE | Price at exit (for exits) |
| `stop_loss` | DOUBLE | Stop loss level |
| `take_profit` | DOUBLE | Take profit level |
| `trade_return` | DOUBLE | Return of the trade (as a decimal, e.g., 0.05 for 5%) |
| `exit_reason` | VARCHAR | Reason for exit (e.g., 'SIGNAL_REVERSAL', 'STOP_LOSS') |
| `metrics` | JSON | Additional metrics and data for the trade |
| `execution_id` | VARCHAR | ID of the backtest execution |
| `created_at` | TIMESTAMP | Timestamp when the trade was logged |

## TradeLogger Class

The `TradeLogger` class in `utils/trade_logger.py` provides a standardized interface for logging trades. It handles:

- Creating the trade log table if it doesn't exist
- Logging individual trades or batches of trades from a DataFrame
- Retrieving trades with filtering options
- Deleting trades or truncating the trade log

### Usage in Backtesting

When implementing a backtest, always use the `TradeLogger` to log trades:

```python
# Import the TradeLogger
from utils.trade_logger import TradeLogger

# Initialize the TradeLogger
db_manager = DBManager()
trade_logger = TradeLogger(db_manager, execution_id="your_execution_id")

# Log a single trade
trade_data = {
    'date': '2023-01-01 10:00:00',
    'symbol': 'SPY',
    'trade_action': 'ENTER_LONG',
    'close': 400.0,
    'entry_price': 400.0,
    'position': 1,
    'rsi_14': 28.5,
    'trend': 'UP'
}
trade_logger.log_trade(strategy_id=123, trade_data=trade_data)

# Log trades from a DataFrame
trade_logger.log_trades_from_dataframe(strategy_id=123, trades_df=results_df, symbol='SPY')
```

### Best Practices

1. **Always use the TradeLogger**: Never write directly to the trade_log table.
2. **Include comprehensive metadata**: Include all relevant information about the trade.
3. **Use consistent trade actions**: Standardize on action names like 'ENTER_LONG', 'EXIT_SHORT', etc.
4. **Set execution_id for each backtest run**: Group trades by execution for easier analysis.
5. **Store technical indicators in metrics**: Include RSI, trend indicators, and other signals in the metrics JSON.

## Trade Logger CLI

The `scripts/trade_logger_cli.py` script provides a command-line interface for working with the trade log:

```bash
# List trades
python scripts/trade_logger_cli.py list --strategy-id 123 --limit 10

# Count trades
python scripts/trade_logger_cli.py count --symbol SPY

# Analyze trades by direction
python scripts/trade_logger_cli.py analyze --by direction

# Export trades to CSV
python scripts/trade_logger_cli.py export --output trades.csv

# Delete trades for a specific strategy
python scripts/trade_logger_cli.py delete --strategy-id 123 --confirm

# Truncate the trade log (delete all trades)
python scripts/trade_logger_cli.py truncate --confirm
```

## Indexes and Performance

The trade log table includes indexes for optimal query performance:

- `trade_id` (primary key)
- `strategy_id` (for filtering by strategy)
- `date` (for time-based queries)
- `execution_id` (for filtering by execution)

## Integration with Metadata System

The trade logging system integrates with the metadata system to track script execution and SQL queries:

- Each trade logging operation is tracked in the metadata tables
- Execution IDs link trades to specific backtest runs
- Strategy IDs link trades to specific strategy configurations

## Trade Analysis

Use the following SQL queries for common trade analysis tasks:

```sql
-- Get all trades for a specific strategy
SELECT * FROM trade_log WHERE strategy_id = 123 ORDER BY date;

-- Calculate win rate by direction
SELECT 
    direction, 
    COUNT(*) as trade_count,
    ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
    ROUND(AVG(trade_return) * 100, 2) as avg_return_pct
FROM trade_log
GROUP BY direction;

-- Calculate performance by exit reason
SELECT 
    exit_reason, 
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 2) as avg_return_pct
FROM trade_log
WHERE exit_reason IS NOT NULL
GROUP BY exit_reason
ORDER BY trade_count DESC;

-- Get top strategies by profit factor
SELECT 
    strategy_id,
    COUNT(*) as trade_count,
    ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
    ROUND(SUM(CASE WHEN trade_return > 0 THEN trade_return ELSE 0 END) / 
          NULLIF(SUM(CASE WHEN trade_return < 0 THEN ABS(trade_return) ELSE 0 END), 0), 2) as profit_factor,
    ROUND(AVG(trade_return) * 100, 2) as avg_return_pct
FROM trade_log
GROUP BY strategy_id
HAVING COUNT(*) >= 5
ORDER BY profit_factor DESC
LIMIT 10;
```

## Extending the System

To extend the trade logging system:

1. Add new columns to the `trade_log` table if needed
2. Update the `TradeLogger` class to handle new fields
3. Create new SQL views or queries for specialized analysis
4. Add new commands to the trade logger CLI

Always maintain backward compatibility and follow the DuckDB-centric approach when extending the system. 