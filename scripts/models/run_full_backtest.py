"""
Full Backtest Runner

This module implements a complete pipeline for running backtests on
multiple strategies, following the DuckDB-centric approach.
"""

import os
import sys
import json
import time
import uuid
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import random
import pandas as pd
import numpy as np

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger
from scripts.models.strategy_generator import generate_strategies, register_strategies_in_metadata
from scripts.models.parameterized_strategy import backtest_parameterized_strategy
from scripts.utils.trade_logger import TradeLogger, get_trade_logger

# Global variables for database connection and logging
db_manager = None
logger = None
trade_logger = None

# The ensure_trade_log_table function is now handled by the TradeLogger class

def run_full_backtest(ticker='SPY', num_strategies=50, start_date=None, end_date=None):
    """
    Run a comprehensive backtest with multiple strategies following DuckDB-centric principles.
    
    Args:
        ticker: Ticker symbol to test
        num_strategies: Number of strategies to test
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
    """
    start_time = time.time()
    
    # Initialize database manager and logger
    global db_manager, logger, trade_logger
    db_manager = DBManager()
    db_manager.connect()
    logger = MetadataLogger(db_manager)
    
    # Generate a unique execution ID for this backtest run
    execution_id = str(uuid.uuid4().hex)
    
    # Initialize the trade logger
    trade_logger = TradeLogger(db_manager, execution_id)
    
    # Log the execution start
    execution_name = f"full_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.log_execution(
        execution_id, 
        execution_name=execution_name,
        description=f"Backtest of {num_strategies} strategies on {ticker}",
        start_time=datetime.now(),
        status="running"
    )
    
    try:
        print("=" * 80)
        print("STEP 1: Setting up database and ensuring tables exist")
        print("=" * 80)
        
        # Ensure all necessary tables exist
        # Trade log table is handled by the TradeLogger
        
        print("\n" + "=" * 80)
        print("STEP 2: Determining date range for backtest")
        print("=" * 80)
        
        if not start_date or not end_date:
            # Get available date range from price data
            date_range = db_manager.conn.execute(f"""
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM price_data_daily
            WHERE symbol = '{ticker}'
            """).fetchone()
            
            if not date_range or not date_range[0] or not date_range[1]:
                raise ValueError(f"No price data found for {ticker}")
                
            if not start_date:
                # Default to one month from the beginning
                start_date = (datetime.strptime(date_range[0], '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')
            
            if not end_date:
                end_date = date_range[1]
        
        print(f"Backtest date range: {start_date} to {end_date}")
        
        print("\n" + "=" * 80)
        print("STEP 3: Generating strategies to test")
        print("=" * 80)
        
        # Generate random strategies
        strategies = generate_strategies(num_strategies)
        
        # Register strategies in metadata
        strategy_ids = register_strategies_in_metadata(db_manager, strategies)
        
        print(f"Generated {len(strategies)} strategies with IDs: {strategy_ids[:5]}...")
        
        print("\n" + "=" * 80)
        print("STEP 4: Loading price data")
        print("=" * 80)
        
        # Ensure we have price data for all timeframes
        timeframes = ['daily', 'hourly', '15min']
        for timeframe in timeframes:
            count = db_manager.conn.execute(f"""
            SELECT COUNT(*) FROM price_data_{timeframe}
            WHERE symbol = '{ticker}'
            AND date BETWEEN '{start_date}' AND '{end_date}'
            """).fetchone()[0]
            
            print(f"Found {count} {timeframe} price records for {ticker}")
            
            if count == 0:
                raise ValueError(f"No {timeframe} price data for {ticker} in the specified date range")
        
        print("\n" + "=" * 80)
        print("STEP 5: Running backtest for each strategy")
        print("=" * 80)
        
        successful_tests = 0
        failed_tests = 0
        successful_strategies = []
        
        for i, (strategy, strategy_id) in enumerate(zip(strategies, strategy_ids)):
            print(f"Testing strategy {strategy_id} ({i+1}/{len(strategies)})...")
            
            try:
                # Run backtest
                print(f"Backtesting strategy {strategy_id} for {ticker} from {start_date} to {end_date}")
                
                results_df, metrics = backtest_parameterized_strategy(
                    strategy, 
                    ticker, 
                    start_date, 
                    end_date,
                    db_manager=db_manager,
                    logger=logger
                )
                
                # Log each trade to the trade_log table using the TradeLogger
                trades = results_df[results_df['trade_action'].notnull()].copy()
                
                # Add symbol to each trade
                trades['symbol'] = ticker
                
                # Log the trades using the TradeLogger
                trade_logger.log_trades_from_dataframe(strategy_id, trades, ticker)
                
                successful_tests += 1
                successful_strategies.append(strategy)
                
                print(f"  Trades: {metrics['total_trades']}, Win Rate: {metrics['win_rate']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.2f}")
                
            except Exception as e:
                print(f"Error testing strategy {strategy_id}: {str(e)}")
                failed_tests += 1
        
        print("\n" + "=" * 80)
        print("STEP 6: Creating summary views for analysis")
        print("=" * 80)
        
        # Create view for trade analysis
        db_manager.conn.execute("""
        CREATE OR REPLACE VIEW trade_analysis AS
        SELECT 
            strategy_id,
            direction,
            COUNT(*) as trade_count,
            SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN trade_return < 0 THEN 1 ELSE 0 END) as losing_trades,
            ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
            ROUND(AVG(trade_return) * 100, 2) as avg_return_pct,
            ROUND(MAX(trade_return) * 100, 2) as max_return_pct,
            ROUND(MIN(trade_return) * 100, 2) as min_return_pct
        FROM trade_log
        GROUP BY strategy_id, direction
        """)
        
        # Create view for overall summary of trade log
        db_manager.conn.execute("""
        CREATE OR REPLACE VIEW trade_log_summary AS
        SELECT 
            COUNT(*) as total_trades,
            COUNT(DISTINCT strategy_id) as total_strategies,
            ROUND(AVG(trade_return) * 100, 2) as avg_return_pct,
            ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
            COUNT(DISTINCT execution_id) as total_executions
        FROM trade_log
        """)
        
        # Create view for all strategies summary
        db_manager.conn.execute("""
        CREATE OR REPLACE VIEW all_strategies_summary AS
        SELECT 
            strategy_id,
            COUNT(*) as total_trades,
            ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
            ROUND(AVG(trade_return) * 100, 2) as avg_return_pct,
            ROUND(SUM(CASE WHEN trade_return > 0 THEN trade_return ELSE 0 END) / 
                NULLIF(SUM(CASE WHEN trade_return < 0 THEN ABS(trade_return) ELSE 0 END), 0), 2) as profit_factor
        FROM trade_log
        GROUP BY strategy_id
        ORDER BY win_rate DESC, avg_return_pct DESC
        """)
        
        # Create view for top strategies
        db_manager.conn.execute("""
        CREATE OR REPLACE VIEW top_strategies AS
        SELECT *
        FROM all_strategies_summary
        WHERE total_trades >= 2
        ORDER BY win_rate DESC, avg_return_pct DESC
        LIMIT 10
        """)
        
        print("\n" + "=" * 80)
        print("STEP 7: Generating summary statistics")
        print("=" * 80)
        
        # Get top strategies by Sharpe ratio
        top_strategies = db_manager.conn.execute("""
        SELECT s.strategy_id, s.total_trades, s.win_rate, s.profit_factor, s.sharpe_ratio, s.annualized_return
        FROM strategy_performance_summary s
        WHERE s.total_trades >= 2 AND s.sharpe_ratio IS NOT NULL
        ORDER BY s.sharpe_ratio DESC
        LIMIT 10
        """).fetchdf()
        
        print("\nTop 10 strategies by Sharpe ratio:")
        print(top_strategies)
        
        # Get trade statistics by direction
        direction_stats = db_manager.conn.execute("""
        SELECT 
            direction,
            COUNT(*) as trade_count,
            ROUND(AVG(trade_return) * 100, 2) as avg_return_pct,
            ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate
        FROM trade_log
        GROUP BY direction
        """).fetchdf()
        
        print("\nTrade statistics by direction:")
        for _, row in direction_stats.iterrows():
            print(f"  {row['direction']}: {row['trade_count']} trades, {row['avg_return_pct']}% avg return, {row['win_rate']}% win rate")
        
        # Get trade statistics by exit reason
        exit_stats = db_manager.conn.execute("""
        SELECT 
            exit_reason,
            COUNT(*) as trade_count,
            ROUND(AVG(trade_return) * 100, 2) as avg_return_pct
        FROM trade_log
        WHERE exit_reason IS NOT NULL
        GROUP BY exit_reason
        ORDER BY trade_count DESC
        """).fetchdf()
        
        print("\nTrade statistics by exit reason:")
        if len(exit_stats) > 0:
            for _, row in exit_stats.iterrows():
                print(f"  {row['exit_reason']}: {row['trade_count']} trades, {row['avg_return_pct']}% avg return")
        else:
            print("  No exit reasons recorded")
        
        # Save analysis queries as templates
        analysis_templates = """
        -- Get all trades for a specific strategy
        SELECT * FROM trade_log WHERE strategy_id = ? ORDER BY date;
        
        -- Get top strategies by win rate
        SELECT * FROM all_strategies_summary WHERE total_trades >= 5 ORDER BY win_rate DESC, avg_return_pct DESC LIMIT 10;
        
        -- Get trade performance by direction
        SELECT direction, COUNT(*) as trade_count, ROUND(AVG(trade_return) * 100, 2) as avg_return_pct, 
        ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate
        FROM trade_log GROUP BY direction;
        
        -- Get trade performance by hour of day
        SELECT json_extract(metrics, '$.hour') as hour, COUNT(*) as trade_count, 
        ROUND(AVG(trade_return) * 100, 2) as avg_return_pct,
        ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate
        FROM trade_log WHERE json_extract(metrics, '$.hour') IS NOT NULL
        GROUP BY hour ORDER BY hour;
        """
        
        os.makedirs('queries', exist_ok=True)
        with open('queries/backtest_analysis_templates.sql', 'w') as f:
            f.write(analysis_templates)
        
        elapsed_time = time.time() - start_time
        
        # Update execution record with completion information
        logger.update_execution(
            execution_id=execution_id,
            end_time=datetime.now(),
            status="completed",
            result_summary=f"Tested {len(strategies)} strategies, found {len(successful_strategies)} successful ones with {trade_logger.get_trades(execution_id=execution_id).shape[0]} trades"
        )
        
        print(f"\nBacktesting completed in {elapsed_time:.2f} seconds")
        print(f"Tested {successful_tests} out of {len(strategies)} strategies successfully")
        print(f"Logged {trade_logger.get_trades(execution_id=execution_id).shape[0]} trades to the database")
        print(f"Analysis queries saved to {os.path.abspath('queries/backtest_analysis_templates.sql')}")
        
        return successful_strategies
    
    except Exception as e:
        print(f"Error in backtest: {str(e)}")
        traceback.print_exc()
        
        # Update execution record with error information
        if 'logger' in locals() and logger and 'execution_id' in locals():
            logger.update_execution(
                execution_id=execution_id,
                end_time=datetime.now(),
                status="failed",
                result_summary=f"Error: {str(e)}"
            )
        
        return None
    finally:
        if db_manager:
            db_manager.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a full backtest with multiple strategies')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol to test')
    parser.add_argument('--num_strategies', type=int, default=50, help='Number of strategies to test')
    parser.add_argument('--start_date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date for backtest (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    run_full_backtest(
        ticker=args.ticker,
        num_strategies=args.num_strategies,
        start_date=args.start_date,
        end_date=args.end_date
    ) 