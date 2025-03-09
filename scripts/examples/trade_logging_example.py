#!/usr/bin/env python
"""
Trade Logging Example

This script demonstrates how to use the Trade Logger system for
capturing and analyzing trades in the DuckDB-centric backtesting framework.
"""

import sys
import json
import uuid
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from scripts.utils.trade_logger import TradeLogger

def create_sample_trades():
    """Create a sample trade DataFrame for demonstration."""
    # Create some sample trades
    trades = []
    
    # Variables for generating consistent trades
    strategy_id = 123
    base_date = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    symbols = ['SPY', 'AAPL', 'MSFT']
    
    # Generate 10 sample trades
    for i in range(10):
        # Alternate between LONG and SHORT
        direction = 'LONG' if i % 2 == 0 else 'SHORT'
        
        # Entry trade
        entry_date = base_date + timedelta(days=i, hours=i % 4)
        entry_price = 100.0 + i
        
        entry_trade = {
            'date': entry_date,
            'symbol': symbols[i % len(symbols)],
            'trade_action': f'ENTER_{direction}',
            'direction': direction,
            'close': entry_price,
            'entry_price': entry_price,
            'position': 1 if direction == 'LONG' else -1,
            'rsi_14': 30 if direction == 'LONG' else 70,
            'trend': 'UP' if direction == 'LONG' else 'DOWN',
            'hour': entry_date.hour,
            'volume': 1000000 + i * 10000,
            'volatility': 0.01 + (i % 5) * 0.001
        }
        trades.append(entry_trade)
        
        # Exit trade (one day later)
        exit_date = entry_date + timedelta(days=1)
        
        # Calculate a return (positive for 70% of trades)
        is_winner = i % 10 < 7  # 70% win rate
        
        if direction == 'LONG':
            exit_price = entry_price * (1.02 if is_winner else 0.99)
            trade_return = (exit_price / entry_price) - 1.0
        else:  # SHORT
            exit_price = entry_price * (0.98 if is_winner else 1.01)
            trade_return = 1.0 - (exit_price / entry_price)
        
        # Determine exit reason
        if i % 3 == 0:
            exit_reason = 'SIGNAL_REVERSAL'
        elif i % 3 == 1:
            exit_reason = 'TAKE_PROFIT' if is_winner else 'STOP_LOSS'
        else:
            exit_reason = 'TIME_EXIT'
        
        exit_trade = {
            'date': exit_date,
            'symbol': symbols[i % len(symbols)],
            'trade_action': f'EXIT_{direction}',
            'direction': direction,
            'close': exit_price,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_return': trade_return,
            'exit_reason': exit_reason,
            'position': 0,
            'rsi_14': 50,
            'trend': 'FLAT',
            'hour': exit_date.hour,
            'volume': 1000000 + i * 10000,
            'volatility': 0.01 + (i % 5) * 0.001
        }
        trades.append(exit_trade)
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)
    
    return trades_df, strategy_id

def demonstrate_trade_logging():
    """Demonstrate the trade logging functionality."""
    print("=" * 60)
    print("TRADE LOGGING EXAMPLE")
    print("=" * 60)
    
    # Initialize the database manager
    db_manager = DBManager()
    
    # Generate a unique execution ID for this example
    execution_id = str(uuid.uuid4().hex)
    print(f"Execution ID: {execution_id}")
    
    # Initialize the trade logger
    trade_logger = TradeLogger(db_manager, execution_id)
    
    print("\n1. Create sample trades")
    trades_df, strategy_id = create_sample_trades()
    print(f"Created {len(trades_df)} sample trades for strategy {strategy_id}")
    
    # Display sample trades
    print("\nSample trades:")
    display_cols = ['date', 'symbol', 'trade_action', 'direction', 'close', 'entry_price', 'exit_price', 'trade_return', 'exit_reason']
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(trades_df[display_cols].head())
    
    print("\n2. Log trades using log_trades_from_dataframe")
    trade_ids = trade_logger.log_trades_from_dataframe(strategy_id, trades_df)
    print(f"Logged {len(trade_ids)} trades with IDs: {trade_ids[:3]}...")
    
    print("\n3. Log a single trade using log_trade")
    single_trade = {
        'date': datetime.now(),
        'symbol': 'GOOGL',
        'trade_action': 'ENTER_LONG',
        'close': 150.0,
        'entry_price': 150.0,
        'position': 1,
        'rsi_14': 25,
        'trend': 'UP',
        'hour': datetime.now().hour,
        'volume': 500000,
        'volatility': 0.015
    }
    trade_id = trade_logger.log_trade(456, single_trade)
    print(f"Logged single trade with ID: {trade_id}")
    
    print("\n4. Retrieve trades for a specific strategy")
    strategy_trades = trade_logger.get_trades(strategy_id=strategy_id)
    print(f"Retrieved {len(strategy_trades)} trades for strategy {strategy_id}")
    print(strategy_trades[display_cols].head(3))
    
    print("\n5. Retrieve trades for a specific symbol")
    symbol_trades = trade_logger.get_trades(symbol='AAPL')
    print(f"Retrieved {len(symbol_trades)} trades for symbol AAPL")
    if not symbol_trades.empty:
        print(symbol_trades[display_cols].head(3))
    
    print("\n6. Analyze trades using SQL")
    
    # Example 1: Win rate by direction
    win_rate_query = """
    SELECT 
        direction, 
        COUNT(*) as trade_count,
        ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
        ROUND(AVG(trade_return) * 100, 2) as avg_return_pct
    FROM trade_log
    WHERE execution_id = ?
    GROUP BY direction
    """
    win_rate_results = db_manager.conn.execute(win_rate_query, [execution_id]).fetchdf()
    
    print("\nWin rate by direction:")
    print(win_rate_results)
    
    # Example 2: Performance by exit reason
    exit_reason_query = """
    SELECT 
        exit_reason, 
        COUNT(*) as trade_count,
        ROUND(AVG(trade_return) * 100, 2) as avg_return_pct
    FROM trade_log
    WHERE exit_reason IS NOT NULL AND execution_id = ?
    GROUP BY exit_reason
    ORDER BY trade_count DESC
    """
    exit_reason_results = db_manager.conn.execute(exit_reason_query, [execution_id]).fetchdf()
    
    print("\nPerformance by exit reason:")
    print(exit_reason_results)
    
    print("\n7. Clean up - delete trades from this example")
    count = trade_logger.delete_trades(execution_id=execution_id)
    print(f"Deleted {count} trades from execution {execution_id}")
    
    # Close the database connection
    db_manager.close()
    
    print("\nTrade logging example completed successfully!")

if __name__ == "__main__":
    try:
        demonstrate_trade_logging()
    except Exception as e:
        print(f"Error in trade logging example: {str(e)}")
        import traceback
        traceback.print_exc() 