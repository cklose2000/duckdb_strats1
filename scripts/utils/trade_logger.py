"""
Trade Logger Utility

This module provides standardized trade logging functionality for the backtesting framework.
It ensures consistent trade capture and logging across all backtesting operations.
"""

import uuid
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import sys
import os
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger

class TradeLogger:
    """
    Standardized trade logging class for all backtesting operations.
    
    This class ensures consistent trade capture, formatting, and logging
    across all backtesting strategies and executions.
    """
    
    def __init__(self, db_manager: Optional[DBManager] = None, execution_id: Optional[str] = None):
        """
        Initialize the trade logger.
        
        Args:
            db_manager: Database manager instance. If None, a new one will be created.
            execution_id: ID of the current backtest execution. If None, a new one will be generated.
        """
        self.db_manager = db_manager if db_manager else DBManager()
        if not hasattr(self.db_manager, 'conn') or self.db_manager.conn is None:
            self.db_manager.connect()
            
        self.execution_id = execution_id if execution_id else str(uuid.uuid4().hex)
        self.metadata_logger = MetadataLogger(self.db_manager)
        
        # Ensure the trade log table exists
        self._ensure_trade_log_table()
    
    def _ensure_trade_log_table(self) -> None:
        """
        Ensure the trade_log table exists in the database.
        """
        # Log this function execution in metadata
        script_id = self.metadata_logger.log_script(
            __file__,
            description="Creates trade log table in the database",
            parameters={},
            script_type="setup"
        )
        
        # Create the trade_log table if it doesn't exist
        self.db_manager.conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            trade_id VARCHAR PRIMARY KEY,
            strategy_id INTEGER,
            date TIMESTAMP,
            symbol VARCHAR,
            action VARCHAR,
            direction VARCHAR,
            price DOUBLE,
            size DOUBLE,
            entry_price DOUBLE,
            exit_price DOUBLE,
            stop_loss DOUBLE,
            take_profit DOUBLE,
            trade_return DOUBLE,
            exit_reason VARCHAR,
            metrics JSON,
            execution_id VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Store the SQL as a query in the metadata
        self.metadata_logger.log_query(
            "CREATE TABLE IF NOT EXISTS trade_log...",
            query_name="create_trade_log_table",
            purpose="To maintain a comprehensive record of all trades with metadata"
        )
        
        # Create index on strategy_id for faster lookups
        self.db_manager.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_log_strategy_id ON trade_log (strategy_id);
        """)
        
        # Create index on date for time-based lookups
        self.db_manager.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_log_date ON trade_log (date);
        """)
        
        # Create index on execution_id for grouping by execution
        self.db_manager.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_log_execution_id ON trade_log (execution_id);
        """)
    
    def log_trade(self, strategy_id: int, trade_data: Dict[str, Any]) -> str:
        """
        Log a trade to the trade_log table.
        
        Args:
            strategy_id: ID of the strategy
            trade_data: Dictionary containing trade data with at least the following keys:
                - date: Timestamp of the trade
                - symbol: Ticker symbol (default: 'SPY')
                - trade_action: Action taken (e.g., 'ENTER_LONG', 'EXIT_SHORT')
                - close: Current price
                - entry_price: Price at entry (for exits)
                - exit_price: Price at exit (for exits)
                - trade_return: Return of the trade (for exits)
                - exit_reason: Reason for exit (for exits)
                
        Returns:
            trade_id: ID of the logged trade
        """
        trade_id = str(uuid.uuid4())
        
        # Extract data from trade_data with defaults
        date = trade_data.get('date')
        symbol = trade_data.get('symbol', 'SPY')
        action = trade_data.get('trade_action')
        
        # Determine direction based on action
        if action and 'LONG' in action:
            direction = 'LONG'
        elif action and 'SHORT' in action:
            direction = 'SHORT'
        else:
            # Default based on position value if available
            position = trade_data.get('position', 0)
            direction = 'LONG' if position > 0 else 'SHORT' if position < 0 else None
        
        price = trade_data.get('close')
        size = trade_data.get('size', 1.0)
        entry_price = trade_data.get('entry_price')
        exit_price = trade_data.get('exit_price')
        trade_return = trade_data.get('trade_return')
        exit_reason = trade_data.get('exit_reason')
        
        # Calculate stop loss and take profit levels if not provided
        stop_loss = trade_data.get('stop_loss')
        take_profit = trade_data.get('take_profit')
        
        if action and 'ENTER' in action and entry_price is not None and (stop_loss is None or take_profit is None):
            # Try to get parameters from strategy_performance_summary
            strategy_params = self.db_manager.conn.execute(
                f"""
                SELECT parameters FROM strategy_performance_summary 
                WHERE strategy_id = {strategy_id}
                """
            ).fetchone()
            
            if strategy_params and strategy_params[0]:
                try:
                    params = json.loads(strategy_params[0])
                    stop_loss_pct = params.get('stop_loss_pct', 0.02)
                    take_profit_pct = params.get('take_profit_pct', 0.05)
                    
                    if direction == 'LONG':
                        if stop_loss is None:
                            stop_loss = round(entry_price * (1 - stop_loss_pct), 2)
                        if take_profit is None:
                            take_profit = round(entry_price * (1 + take_profit_pct), 2)
                    else:  # SHORT
                        if stop_loss is None:
                            stop_loss = round(entry_price * (1 + stop_loss_pct), 2)
                        if take_profit is None:
                            take_profit = round(entry_price * (1 - take_profit_pct), 2)
                except (json.JSONDecodeError, TypeError):
                    # If parameters can't be parsed, use defaults
                    pass
        
        # Store additional trade metrics
        metrics = {
            'position': trade_data.get('position'),
            'rsi': trade_data.get('rsi_14'),
            'trend': trade_data.get('trend'),
            'hour': trade_data.get('hour'),
            'volume': trade_data.get('volume'),
            'volatility': trade_data.get('volatility')
        }
        
        # Add any other metrics from trade_data not already included
        for key, value in trade_data.items():
            if key not in ['date', 'symbol', 'trade_action', 'close', 'entry_price', 
                          'exit_price', 'trade_return', 'exit_reason', 'position', 
                          'rsi_14', 'trend', 'hour', 'volume', 'volatility', 'size', 
                          'stop_loss', 'take_profit'] and value is not None:
                metrics[key] = value
        
        # Insert the trade
        self.db_manager.conn.execute(
            """
            INSERT INTO trade_log (
                trade_id, strategy_id, date, symbol, action, direction, price,
                size, entry_price, exit_price, stop_loss, take_profit, 
                trade_return, exit_reason, metrics, execution_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id, strategy_id, date, symbol, action, direction, price,
                size, entry_price, exit_price, stop_loss, take_profit,
                trade_return, exit_reason, json.dumps(metrics), self.execution_id
            )
        )
        
        return trade_id
    
    def log_trades_from_dataframe(self, strategy_id: int, trades_df: pd.DataFrame, symbol: str = 'SPY') -> List[str]:
        """
        Log multiple trades from a DataFrame to the trade_log table.
        
        Args:
            strategy_id: ID of the strategy
            trades_df: DataFrame containing trade data with at least 'trade_action' column
            symbol: Ticker symbol (default: 'SPY')
            
        Returns:
            List of trade IDs for the logged trades
        """
        trade_ids = []
        
        # Filter only rows with trade_action
        if 'trade_action' in trades_df.columns:
            trades = trades_df[trades_df['trade_action'].notnull()].copy()
            
            for idx, trade in trades.iterrows():
                # Convert pandas Series to dict
                trade_data = trade.to_dict()
                
                # Add symbol if not present
                if 'symbol' not in trade_data:
                    trade_data['symbol'] = symbol
                
                # Log the trade
                trade_id = self.log_trade(strategy_id, trade_data)
                trade_ids.append(trade_id)
        
        return trade_ids
    
    def get_trades(self, strategy_id: Optional[int] = None, 
                  execution_id: Optional[str] = None,
                  symbol: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve trades from the trade_log table.
        
        Args:
            strategy_id: Filter by strategy ID
            execution_id: Filter by execution ID
            symbol: Filter by symbol
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            
        Returns:
            DataFrame containing trade data
        """
        query = "SELECT * FROM trade_log WHERE 1=1"
        params = []
        
        if strategy_id is not None:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        
        if execution_id is not None:
            query += " AND execution_id = ?"
            params.append(execution_id)
        
        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date is not None:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date is not None:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        return self.db_manager.conn.execute(query, params).fetchdf()
    
    def delete_trades(self, strategy_id: Optional[int] = None, 
                     execution_id: Optional[str] = None) -> int:
        """
        Delete trades from the trade_log table.
        
        Args:
            strategy_id: Filter by strategy ID
            execution_id: Filter by execution ID
            
        Returns:
            Number of trades deleted
        """
        query = "DELETE FROM trade_log WHERE 1=1"
        params = []
        
        if strategy_id is not None:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        
        if execution_id is not None:
            query += " AND execution_id = ?"
            params.append(execution_id)
        
        # Get the count first
        count_query = query.replace("DELETE FROM", "SELECT COUNT(*) FROM")
        count = self.db_manager.conn.execute(count_query, params).fetchone()[0]
        
        # Then delete
        self.db_manager.conn.execute(query, params)
        
        return count
    
    def truncate_trade_log(self) -> None:
        """
        Truncate the trade_log table, removing all trades.
        """
        self.db_manager.conn.execute("DELETE FROM trade_log")
        print("Trade log table truncated.")
    
    def close(self) -> None:
        """
        Close the database connection.
        """
        if self.db_manager and hasattr(self.db_manager, 'close'):
            self.db_manager.close()


# Singleton instance for easy access
_trade_logger_instance = None

def get_trade_logger(db_manager: Optional[DBManager] = None, 
                   execution_id: Optional[str] = None) -> TradeLogger:
    """
    Get or create a TradeLogger instance.
    
    Args:
        db_manager: Database manager instance
        execution_id: ID of the current backtest execution
        
    Returns:
        TradeLogger instance
    """
    global _trade_logger_instance
    
    if _trade_logger_instance is None:
        _trade_logger_instance = TradeLogger(db_manager, execution_id)
    
    return _trade_logger_instance 