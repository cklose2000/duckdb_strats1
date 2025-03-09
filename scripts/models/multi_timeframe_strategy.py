"""
Multi-Timeframe Strategy Backtesting

This script implements a simple multi-timeframe strategy and backtests it using
the SPY data at different timeframes. All computations are done with SQL in DuckDB.
"""

import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path to import from utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super(DateTimeEncoder, self).default(obj)

def backtest_multi_timeframe_strategy(ticker='SPY', start_date=None, end_date=None, 
                                     db_manager=None, logger=None):
    """
    Backtest a multi-timeframe strategy using daily, hourly, and 5-minute data.
    
    The strategy uses:
    - Daily data for trend identification (SMA crossovers)
    - Hourly data for entry timing (RSI and volatility)
    - 5-minute data for precise entries and exits
    
    Args:
        ticker (str): Ticker symbol to backtest.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        db_manager (DBManager, optional): Database manager instance.
        logger (MetadataLogger, optional): Metadata logger instance.
        
    Returns:
        dict: Backtest results.
    """
    # Create DB manager and logger if not provided
    if db_manager is None:
        db_manager = DBManager()
        db_manager.connect()
        
    if logger is None:
        logger = MetadataLogger(db_manager.conn)
    
    # Convert date parameters to strings if they are datetime objects
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
    
    # Log script execution
    script_id = logger.log_script(
        __file__,
        description="Backtest a multi-timeframe trading strategy",
        parameters={
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date
        },
        script_type="model",
        dependencies=["price_data_daily", "price_data_60min", "price_data_5min"]
    )
    
    start_time = time.time()
    
    try:
        # Determine date range if not provided
        if not start_date or not end_date:
            date_range_query = """
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM price_data_daily
            WHERE ticker = ?
            """
            min_date, max_date = db_manager.conn.execute(date_range_query, (ticker,)).fetchone()
            
            if not start_date:
                start_date = min_date.strftime('%Y-%m-%d') if isinstance(min_date, datetime) else min_date
            if not end_date:
                end_date = max_date.strftime('%Y-%m-%d') if isinstance(max_date, datetime) else max_date
                
        print(f"Backtesting {ticker} from {start_date} to {end_date}")
        
        # Step 1: Create a daily signals table with trend indicators
        daily_signals_query = """
        WITH daily_data AS (
            SELECT
                ticker,
                date,
                open,
                high,
                low,
                close,
                volume,
                -- Simple Moving Averages
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma_50,
                -- Volatility
                STDDEV(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) / 
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volatility_20,
                -- Price returns
                close / LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return
            FROM price_data_daily
            WHERE ticker = ?
            AND date BETWEEN ? AND ?
        )
        SELECT
            ticker,
            date,
            close,
            sma_20,
            sma_50,
            volatility_20,
            daily_return,
            -- Trend signals
            CASE 
                WHEN close > sma_20 AND sma_20 > sma_50 THEN 'UPTREND'
                WHEN close < sma_20 AND sma_20 < sma_50 THEN 'DOWNTREND'
                ELSE 'NEUTRAL'
            END AS trend,
            -- Trend change signals
            CASE
                WHEN close > sma_20 AND LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) <= LAG(sma_20, 1) OVER (PARTITION BY ticker ORDER BY date) THEN 'BUY_SIGNAL'
                WHEN close < sma_20 AND LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) >= LAG(sma_20, 1) OVER (PARTITION BY ticker ORDER BY date) THEN 'SELL_SIGNAL'
                ELSE NULL
            END AS signal
        FROM daily_data
        ORDER BY date
        """
        
        # Execute the daily signals query and create a temporary table
        db_manager.conn.execute("DROP TABLE IF EXISTS temp_daily_signals")
        db_manager.conn.execute(
            f"CREATE TABLE temp_daily_signals AS {daily_signals_query}",
            (ticker, start_date, end_date)
        )
        
        # Step 2: Add hourly indicators for entry timing
        hourly_signals_query = """
        WITH hourly_data AS (
            SELECT
                ticker,
                date,
                open,
                high,
                low,
                close,
                volume,
                -- Cast date to date for joining with daily signals
                CAST(date AS DATE) AS trade_date,
                -- RSI calculation components
                close - LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) AS price_change,
                -- Hour of day for timing
                EXTRACT(HOUR FROM date) AS hour
            FROM price_data_60min
            WHERE ticker = ?
            AND date BETWEEN ? AND ?
        ),
        hourly_rsi AS (
            SELECT
                ticker,
                date,
                trade_date,
                hour,
                close,
                price_change,
                -- Calculate gains and losses for RSI
                CASE WHEN price_change > 0 THEN price_change ELSE 0 END AS gain,
                CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END AS loss,
                -- Calculate average gain and average loss over 14 periods
                AVG(CASE WHEN price_change > 0 THEN price_change ELSE 0 END) OVER (
                    PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) AS avg_gain,
                AVG(CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END) OVER (
                    PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) AS avg_loss
            FROM hourly_data
        )
        SELECT
            h.ticker,
            h.date,
            h.trade_date,
            h.hour,
            h.close,
            -- Calculate RSI
            CASE 
                WHEN h.avg_loss = 0 THEN 100
                ELSE 100 - (100 / (1 + (h.avg_gain / NULLIF(h.avg_loss, 0))))
            END AS rsi_14,
            -- Get daily trend
            d.trend,
            d.signal AS daily_signal
        FROM hourly_rsi h
        LEFT JOIN temp_daily_signals d ON h.trade_date = CAST(d.date AS DATE) AND h.ticker = d.ticker
        ORDER BY h.date
        """
        
        # Execute the hourly signals query and create a temporary table
        db_manager.conn.execute("DROP TABLE IF EXISTS temp_hourly_signals")
        db_manager.conn.execute(
            f"CREATE TABLE temp_hourly_signals AS {hourly_signals_query}",
            (ticker, start_date, end_date)
        )
        
        # Step 3: Generate trading signals from the combined data
        trading_signals_query = """
        WITH entry_signals AS (
            SELECT
                date,
                close,
                trend,
                daily_signal,
                rsi_14,
                hour,
                -- Generate entry signals based on trend and RSI
                CASE
                    WHEN trend = 'UPTREND' AND rsi_14 < 30 THEN 'BUY'
                    WHEN trend = 'DOWNTREND' AND rsi_14 > 70 THEN 'SELL'
                    WHEN daily_signal = 'BUY_SIGNAL' AND rsi_14 < 40 THEN 'BUY'
                    WHEN daily_signal = 'SELL_SIGNAL' AND rsi_14 > 60 THEN 'SELL'
                    ELSE NULL
                END AS signal,
                -- Add lag signals for detecting changes
                LAG(
                    CASE
                        WHEN trend = 'UPTREND' AND rsi_14 < 30 THEN 'BUY'
                        WHEN trend = 'DOWNTREND' AND rsi_14 > 70 THEN 'SELL'
                        WHEN daily_signal = 'BUY_SIGNAL' AND rsi_14 < 40 THEN 'BUY'
                        WHEN daily_signal = 'SELL_SIGNAL' AND rsi_14 > 60 THEN 'SELL'
                        ELSE NULL
                    END, 1
                ) OVER (ORDER BY date) AS prev_signal
            FROM temp_hourly_signals
        )
        SELECT
            date,
            close,
            trend,
            rsi_14,
            hour,
            signal,
            -- Only take trades when signal changes
            CASE
                WHEN signal = 'BUY' AND (prev_signal IS NULL OR prev_signal != 'BUY') THEN 'ENTER_LONG'
                WHEN signal = 'SELL' AND (prev_signal IS NULL OR prev_signal != 'SELL') THEN 'ENTER_SHORT'
                ELSE NULL
            END AS trade_action
        FROM entry_signals
        ORDER BY date
        """
        
        # Execute the trading signals query
        trading_signals = db_manager.query_to_df(trading_signals_query)
        
        # Step 4: Simulate the trades
        def simulate_trades(signals_df):
            signals_df = signals_df.copy()
            
            # Add columns for trade tracking
            signals_df['position'] = 0
            signals_df['entry_price'] = None
            signals_df['exit_price'] = None
            signals_df['trade_return'] = 0.0
            signals_df['cumulative_return'] = 1.0
            
            position = 0
            entry_price = 0
            trade_pnl = []
            
            for i, row in signals_df.iterrows():
                if position == 0:  # No position
                    if row['trade_action'] == 'ENTER_LONG':
                        position = 1
                        entry_price = row['close']
                        signals_df.at[i, 'position'] = position
                        signals_df.at[i, 'entry_price'] = entry_price
                    elif row['trade_action'] == 'ENTER_SHORT':
                        position = -1
                        entry_price = row['close']
                        signals_df.at[i, 'position'] = position
                        signals_df.at[i, 'entry_price'] = entry_price
                
                elif position == 1:  # Long position
                    signals_df.at[i, 'position'] = position
                    if row['trade_action'] == 'ENTER_SHORT':
                        # Exit long and enter short
                        exit_price = row['close']
                        trade_return = (exit_price / entry_price) - 1
                        trade_pnl.append((entry_price, exit_price, trade_return))
                        signals_df.at[i, 'exit_price'] = exit_price
                        signals_df.at[i, 'trade_return'] = trade_return
                        
                        # Enter short
                        position = -1
                        entry_price = row['close']
                        signals_df.at[i, 'position'] = position
                        signals_df.at[i, 'entry_price'] = entry_price
                
                elif position == -1:  # Short position
                    signals_df.at[i, 'position'] = position
                    if row['trade_action'] == 'ENTER_LONG':
                        # Exit short and enter long
                        exit_price = row['close']
                        trade_return = 1 - (exit_price / entry_price)
                        trade_pnl.append((entry_price, exit_price, trade_return))
                        signals_df.at[i, 'exit_price'] = exit_price
                        signals_df.at[i, 'trade_return'] = trade_return
                        
                        # Enter long
                        position = 1
                        entry_price = row['close']
                        signals_df.at[i, 'position'] = position
                        signals_df.at[i, 'entry_price'] = entry_price
            
            # Close any open position at the end
            if position != 0:
                last_price = signals_df.iloc[-1]['close']
                if position == 1:
                    trade_return = (last_price / entry_price) - 1
                else:  # position == -1
                    trade_return = 1 - (last_price / entry_price)
                trade_pnl.append((entry_price, last_price, trade_return))
                signals_df.iloc[-1, signals_df.columns.get_loc('exit_price')] = last_price
                signals_df.iloc[-1, signals_df.columns.get_loc('trade_return')] = trade_return
            
            # Calculate cumulative returns
            cum_return = 1.0
            for i, row in signals_df.iterrows():
                if row['trade_return'] != 0:
                    cum_return *= (1 + row['trade_return'])
                signals_df.at[i, 'cumulative_return'] = cum_return
            
            # Calculate strategy metrics
            total_trades = len(trade_pnl)
            win_trades = sum(1 for _, _, ret in trade_pnl if ret > 0)
            loss_trades = sum(1 for _, _, ret in trade_pnl if ret <= 0)
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            avg_win = sum(ret for _, _, ret in trade_pnl if ret > 0) / win_trades if win_trades > 0 else 0
            avg_loss = sum(ret for _, _, ret in trade_pnl if ret <= 0) / loss_trades if loss_trades > 0 else 0
            profit_factor = abs(sum(ret for _, _, ret in trade_pnl if ret > 0) / sum(ret for _, _, ret in trade_pnl if ret < 0)) if sum(ret for _, _, ret in trade_pnl if ret < 0) != 0 else float('inf')
            
            # Convert dates to strings for JSON serialization
            start_date = signals_df.iloc[0]['date'].strftime('%Y-%m-%d') if len(signals_df) > 0 else None
            end_date = signals_df.iloc[-1]['date'].strftime('%Y-%m-%d') if len(signals_df) > 0 else None
            
            metrics = {
                'total_trades': total_trades,
                'win_trades': win_trades,
                'loss_trades': loss_trades,
                'win_rate': float(win_rate),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.99,
                'final_return': float(cum_return - 1),
                'annualized_return': float((cum_return - 1) / ((signals_df.iloc[-1]['date'] - signals_df.iloc[0]['date']).days / 365)) if len(signals_df) > 1 else 0.0,
                'start_date': start_date,
                'end_date': end_date
            }
            
            return signals_df, metrics
        
        # Run the simulation
        results_df, metrics = simulate_trades(trading_signals)
        
        # Step 5: Save the results to the database
        db_manager.conn.execute("DROP TABLE IF EXISTS multi_timeframe_backtest_results")
        
        db_manager.conn.register("temp_results", results_df)
        
        create_results_table_query = """
        CREATE TABLE multi_timeframe_backtest_results AS
        SELECT * FROM temp_results
        """
        db_manager.execute(create_results_table_query)
        
        # Log model/strategy
        model_id = logger.log_model(
            model_name="multi_timeframe_strategy",
            model_type="Trading Strategy",
            description="Multi-timeframe trading strategy using daily trend and hourly RSI",
            parameters={
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date
            },
            performance_metrics=metrics,
            script_id=script_id
        )
        
        # Log model training
        logger.log_model_training(
            model_id=model_id,
            training_time=time.time() - start_time,
            performance_metrics=metrics
        )
        
        # Generate strategy performance plots
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Cumulative returns
        plt.subplot(2, 1, 1)
        plt.plot(results_df['date'], results_df['cumulative_return'], label='Strategy Returns')
        plt.title(f'Multi-Timeframe Strategy Performance - {ticker}')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Trades
        plt.subplot(2, 1, 2)
        plt.plot(results_df['date'], results_df['close'], label='Price')
        
        # Mark buy and sell signals
        buys = results_df[results_df['trade_action'] == 'ENTER_LONG']
        sells = results_df[results_df['trade_action'] == 'ENTER_SHORT']
        
        plt.scatter(buys['date'], buys['close'], color='green', marker='^', s=100, label='Buy')
        plt.scatter(sells['date'], sells['close'], color='red', marker='v', s=100, label='Sell')
        
        plt.title(f'Trade Signals - {ticker}')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        image_path = str(Path(__file__).parent.parent.parent / "notebooks" / f"{ticker}_multi_timeframe_strategy.png")
        plt.savefig(image_path)
        plt.close()
        
        # Print the performance metrics
        print("\nStrategy Performance Metrics:")
        print(f"Total trades: {metrics['total_trades']}")
        print(f"Win rate: {metrics['win_rate']:.2%}")
        print(f"Profit factor: {metrics['profit_factor']:.2f}")
        print(f"Total return: {metrics['final_return']:.2%}")
        print(f"Annualized return: {metrics['annualized_return']:.2%}")
        
        # Log execution success
        execution_time = time.time() - start_time
        logger.log_script_execution(script_id, execution_time, "success")
        
        # Return the results
        return {
            'model_id': model_id,
            'metrics': metrics,
            'image_path': image_path
        }
        
    except Exception as e:
        # Log execution failure
        execution_time = time.time() - start_time
        logger.log_script_execution(script_id, execution_time, f"failed: {str(e)}")
        raise
    finally:
        # Clean up temporary tables
        db_manager.conn.execute("DROP TABLE IF EXISTS temp_daily_signals")
        db_manager.conn.execute("DROP TABLE IF EXISTS temp_hourly_signals")

if __name__ == "__main__":
    # Create DB manager and logger
    db_manager = DBManager()
    db_manager.connect()
    logger = MetadataLogger(db_manager.conn)
    
    try:
        # Log execution start
        execution_id = logger.log_execution(
            execution_name="multi_timeframe_backtest",
            description="Backtest multi-timeframe trading strategy on SPY",
            parameters=json.dumps({
                "ticker": "SPY",
                "start_date": None,
                "end_date": None
            }, cls=DateTimeEncoder)
        )
        
        # Run the backtest
        results = backtest_multi_timeframe_strategy(
            ticker="SPY",
            db_manager=db_manager,
            logger=logger
        )
        
        # Ensure all results are JSON serializable
        serializable_metrics = json.loads(json.dumps(results['metrics'], cls=DateTimeEncoder))
        
        # Log execution completion
        logger.log_execution_end(
            execution_id,
            "completed",
            "Multi-timeframe strategy backtest completed successfully",
            serializable_metrics
        )
        
        print(f"\nResults saved to database and plot saved to {results['image_path']}")
        
    except Exception as e:
        print(f"Error backtesting multi-timeframe strategy: {e}")
        
        # Log execution failure
        if 'execution_id' in locals():
            logger.log_execution_end(
                execution_id,
                "failed",
                f"Error: {str(e)}"
            )
    finally:
        # Close the database connection
        db_manager.close() 