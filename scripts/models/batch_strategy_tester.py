"""
Batch Strategy Tester

This script generates and tests multiple strategy variants in a batch process.
It creates 100 random strategy variants, backtests them, and selects the top
performers based on specified metrics.
"""

import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse
import sys
import traceback

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger
from scripts.models.strategy_generator import StrategyParameters, generate_strategies, register_strategies_in_metadata
from scripts.models.parameterized_strategy import backtest_parameterized_strategy

def get_date_range_from_db(db_manager=None):
    """
    Get the min and max dates for SPY 5-minute data from the database.
    
    Args:
        db_manager (DBManager, optional): Database manager instance
        
    Returns:
        tuple: (start_date, end_date) as strings in 'YYYY-MM-DD' format
    """
    if db_manager is None:
        db_manager = DBManager()
        db_manager.connect()
    
    date_query = """
    SELECT MIN(date) as min_date, MAX(date) as max_date
    FROM price_data_5min
    WHERE ticker = 'SPY'
    """
    min_date, max_date = db_manager.conn.execute(date_query).fetchone()
    
    # Format dates as strings
    start_date = min_date.strftime('%Y-%m-%d')
    end_date = max_date.strftime('%Y-%m-%d')
    
    return start_date, end_date

def create_strategy_summary_table(db_manager=None):
    """
    Create a summary table for strategy performance metrics.
    
    Args:
        db_manager (DBManager, optional): Database manager instance
        
    Returns:
        bool: True if successful
    """
    # Create DB manager if not provided
    if db_manager is None:
        from utils.db_manager import DBManager
        db_manager = DBManager()
        db_manager.connect()
    
    # Create the summary table
    db_manager.conn.execute("""
    CREATE TABLE IF NOT EXISTS strategy_performance_summary (
        strategy_id INTEGER PRIMARY KEY,
        total_trades INTEGER,
        win_rate DOUBLE,
        profit_factor DOUBLE,
        avg_return DOUBLE,
        max_drawdown DOUBLE,
        sharpe_ratio DOUBLE,
        annualized_return DOUBLE,
        parameters JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create a view that combines the backtest results with the summary
    db_manager.conn.execute("""
    CREATE OR REPLACE VIEW strategy_backtest_results AS
    SELECT 
        r.*,
        s.win_rate,
        s.profit_factor,
        s.sharpe_ratio
    FROM parameterized_backtest_results r
    JOIN strategy_performance_summary s ON r.strategy_id = s.strategy_id
    """)
    
    return True

def save_strategy_metrics(strategy_id, metrics, parameters, db_manager=None):
    """
    Save strategy performance metrics to the summary table.
    
    Args:
        strategy_id (int): ID of the strategy
        metrics (dict): Performance metrics
        parameters (dict): Strategy parameters
        db_manager (DBManager, optional): Database manager instance
    """
    if db_manager is None:
        db_manager = DBManager()
        db_manager.connect()
    
    # Check if metrics already exist for this strategy
    existing = db_manager.conn.execute(
        "SELECT COUNT(*) FROM strategy_performance_summary WHERE strategy_id = ?", 
        (strategy_id,)
    ).fetchone()[0]
    
    if existing > 0:
        # Update existing record
        db_manager.conn.execute("""
        UPDATE strategy_performance_summary
        SET total_trades = ?,
            win_rate = ?,
            profit_factor = ?,
            avg_return = ?,
            max_drawdown = ?,
            sharpe_ratio = ?,
            annualized_return = ?,
            parameters = ?
        WHERE strategy_id = ?
        """, (
            metrics["total_trades"],
            metrics["win_rate"],
            metrics["profit_factor"],
            metrics["avg_return"],
            metrics["max_drawdown"],
            metrics["sharpe_ratio"],
            metrics["annualized_return"],
            json.dumps(parameters),
            strategy_id
        ))
    else:
        # Insert new record
        db_manager.conn.execute("""
        INSERT INTO strategy_performance_summary (
            strategy_id,
            total_trades,
            win_rate,
            profit_factor,
            avg_return,
            max_drawdown,
            sharpe_ratio,
            annualized_return,
            parameters
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy_id,
            metrics["total_trades"],
            metrics["win_rate"],
            metrics["profit_factor"],
            metrics["avg_return"],
            metrics["max_drawdown"],
            metrics["sharpe_ratio"],
            metrics["annualized_return"],
            json.dumps(parameters)
        ))

def get_top_strategies(num_strategies=10, sort_by="sharpe_ratio", db_manager=None):
    """
    Get the top performing strategies based on a specified metric.
    
    Args:
        num_strategies (int): Number of top strategies to retrieve
        sort_by (str): Metric to sort by (sharpe_ratio, profit_factor, win_rate, etc.)
        db_manager (DBManager, optional): Database manager instance
        
    Returns:
        pd.DataFrame: DataFrame of top strategies with their metrics
    """
    if db_manager is None:
        db_manager = DBManager()
        db_manager.connect()
    
    # Query for top strategies
    query = f"""
    SELECT 
        strategy_id,
        total_trades,
        win_rate,
        profit_factor,
        avg_return,
        max_drawdown,
        sharpe_ratio,
        annualized_return,
        parameters
    FROM strategy_performance_summary
    WHERE total_trades > 0
    ORDER BY {sort_by} DESC
    LIMIT ?
    """
    
    # Execute the query and convert to DataFrame
    results = db_manager.conn.execute(query, (num_strategies,)).fetchall()
    columns = [
        "strategy_id", "total_trades", "win_rate", "profit_factor", 
        "avg_return", "max_drawdown", "sharpe_ratio", "annualized_return", "parameters"
    ]
    
    return pd.DataFrame(results, columns=columns)

def batch_test_strategies(num_strategies=100, ticker='SPY', plot_top=10):
    """
    Generate and backtest multiple strategy variants.
    
    Args:
        num_strategies (int): Number of strategies to generate and test
        ticker (str): Ticker symbol to test
        plot_top (int): Number of top strategies to generate plots for
        
    Returns:
        pd.DataFrame: DataFrame of all strategy results sorted by Sharpe ratio
    """
    print(f"Starting batch testing of {num_strategies} strategy variants for {ticker}")
    
    # Initialize database connections
    db_manager = DBManager()
    db_manager.connect()
    logger = MetadataLogger(db_manager.conn)
    
    # Log the batch testing execution
    batch_id = logger.log_script(
        __file__,
        description=f"Batch testing {num_strategies} strategy variants",
        parameters={
            "num_strategies": num_strategies,
            "ticker": ticker
        },
        script_type="batch_testing"
    )
    
    start_time = time.time()
    
    try:
        # Get date range from database
        start_date, end_date = get_date_range_from_db(db_manager)
        
        # Generate strategies
        print(f"Generating {num_strategies} random strategy variants...")
        strategies = generate_strategies(num_strategies)
        
        # Register strategies in metadata
        strategy_metadata_ids = register_strategies_in_metadata(
            strategies, db_manager, logger
        )
        
        # Create summary table
        create_strategy_summary_table(db_manager)
        
        # Backtest each strategy
        successful_strategies = 0
        
        for i, strategy in enumerate(strategies):
            print(f"Testing strategy {strategy.strategy_id} ({i+1}/{num_strategies})...")
            try:
                # Run the backtest
                results_df, metrics = backtest_parameterized_strategy(
                    strategy,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    db_manager=db_manager,
                    logger=logger,
                    plot_returns=False  # Don't plot all strategies
                )
                
                # Save the metrics
                save_strategy_metrics(
                    strategy.strategy_id,
                    metrics,
                    strategy.to_dict(),
                    db_manager
                )
                
                successful_strategies += 1
                
                # Print some basic metrics
                print(f"  Trades: {metrics['total_trades']}, Win Rate: {metrics['win_rate']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.2f}")
                
            except Exception as e:
                print(f"Error testing strategy {strategy.strategy_id}: {str(e)}")
                traceback.print_exc()
        
        # Get top strategies by Sharpe ratio
        top_by_sharpe = get_top_strategies(
            num_strategies=plot_top, 
            sort_by="sharpe_ratio",
            db_manager=db_manager
        )
        
        print("\nTop strategies by Sharpe ratio:")
        print(top_by_sharpe[["strategy_id", "total_trades", "win_rate", "sharpe_ratio", "profit_factor"]])
        
        # Get top strategies by profit factor
        top_by_profit = get_top_strategies(
            num_strategies=plot_top, 
            sort_by="profit_factor",
            db_manager=db_manager
        )
        
        print("\nTop strategies by profit factor:")
        print(top_by_profit[["strategy_id", "total_trades", "win_rate", "sharpe_ratio", "profit_factor"]])
        
        # Create plots for top strategies
        print(f"\nGenerating plots for top {plot_top} strategies...")
        
        for _, row in top_by_sharpe.iterrows():
            strategy_id = row["strategy_id"]
            parameters = json.loads(row["parameters"])
            
            # Recreate the strategy object
            strategy = StrategyParameters(
                strategy_id=parameters["strategy_id"],
                sma_fast_period=parameters["sma_fast_period"],
                sma_slow_period=parameters["sma_slow_period"],
                rsi_period=parameters["rsi_period"],
                rsi_buy_threshold=parameters["rsi_buy_threshold"],
                rsi_sell_threshold=parameters["rsi_sell_threshold"],
                rsi_buy_signal_threshold=parameters["rsi_buy_signal_threshold"],
                rsi_sell_signal_threshold=parameters["rsi_sell_signal_threshold"],
                volatility_period=parameters["volatility_period"]
            )
            
            # Run the backtest with plot generation
            try:
                backtest_parameterized_strategy(
                    strategy,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    db_manager=db_manager,
                    logger=logger,
                    plot_returns=True
                )
            except Exception as e:
                print(f"Error generating plot for strategy {strategy_id}: {str(e)}")
        
        # Create a summary query in the database
        summary_query = """
        CREATE OR REPLACE VIEW strategy_ranking AS
        SELECT 
            strategy_id,
            total_trades,
            win_rate,
            profit_factor,
            sharpe_ratio,
            annualized_return,
            parameters,
            RANK() OVER (ORDER BY sharpe_ratio DESC) AS sharpe_rank,
            RANK() OVER (ORDER BY profit_factor DESC) AS profit_factor_rank,
            RANK() OVER (ORDER BY win_rate DESC) AS win_rate_rank,
            RANK() OVER (ORDER BY annualized_return DESC) AS return_rank
        FROM strategy_performance_summary
        WHERE total_trades >= 5
        """
        
        db_manager.conn.execute(summary_query)
        
        # Log completion
        elapsed_time = time.time() - start_time
        
        logger.log_execution_end(
            batch_id,
            status="success",
            result_summary=f"Successfully tested {successful_strategies} strategies",
            result_metrics=json.dumps({
                "strategies_tested": successful_strategies,
                "top_strategy_id": int(top_by_sharpe.iloc[0]["strategy_id"]) if not top_by_sharpe.empty else None,
                "top_sharpe": top_by_sharpe.iloc[0]["sharpe_ratio"] if not top_by_sharpe.empty else 0,
                "elapsed_time": elapsed_time
            })
        )
        
        print(f"\nBatch testing complete! Tested {successful_strategies} strategies in {elapsed_time:.2f} seconds")
        
        # Return all strategy results
        all_strategies = get_top_strategies(
            num_strategies=num_strategies,
            sort_by="sharpe_ratio",
            db_manager=db_manager
        )
        
        return all_strategies
        
    except Exception as e:
        # Log error
        logger.log_execution_end(
            batch_id,
            status="error",
            result_summary=f"Error in batch testing: {str(e)}",
            result_metrics=json.dumps({"error": str(e)})
        )
        print(f"Error in batch testing: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame on error
    finally:
        # Close database connection
        db_manager.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch test multiple trading strategies')
    parser.add_argument('--num', type=int, default=100, help='Number of strategies to test')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol to test')
    parser.add_argument('--plot-top', type=int, default=10, help='Number of top strategies to plot')
    
    args = parser.parse_args()
    
    # Run the batch testing
    batch_test_strategies(
        num_strategies=args.num,
        ticker=args.ticker,
        plot_top=args.plot_top
    ) 