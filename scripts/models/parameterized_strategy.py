"""
Parameterized Multi-Timeframe Strategy

This module implements a parameterized version of the multi-timeframe strategy,
allowing it to be run with different parameters for automated backtesting.
"""

import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path to import from utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger
from scripts.models.strategy_generator import StrategyParameters

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super(DateTimeEncoder, self).default(obj)

def calculate_metrics(returns):
    """
    Calculate performance metrics for a strategy.
    
    Args:
        returns (pd.Series): Series of trade returns
        
    Returns:
        dict: Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "annualized_return": 0.0
        }
    
    # Basic stats
    win_rate = (returns > 0).mean() * 100
    
    # Profit factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) * 100
    
    # Sharpe ratio (assuming 252 trading days in a year)
    risk_free_rate = 0.02 / 252  # 2% annual risk-free rate converted to daily
    excess_returns = returns - risk_free_rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() != 0 else 0
    
    # Annualized return
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    
    return {
        "total_trades": len(returns),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return": returns.mean() * 100,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "annualized_return": annualized_return * 100
    }

def backtest_parameterized_strategy(strategy_params, ticker='SPY', start_date=None, end_date=None, 
                                    db_manager=None, logger=None, plot_returns=False):
    """
    Backtest a parameterized trading strategy.
    
    Args:
        strategy_params (StrategyParameters): Parameters for the strategy
        ticker (str): Ticker symbol to backtest
        start_date (str): Start date for backtest (YYYY-MM-DD)
        end_date (str): End date for backtest (YYYY-MM-DD)
        db_manager (DBManager, optional): Database manager instance
        logger (MetadataLogger, optional): Logger instance
        plot_returns (bool): Whether to plot the returns
        
    Returns:
        tuple: (results_df, metrics) - DataFrame with results and dict with metrics
    """
    start_time = time.time()
    
    # Create DB manager if not provided
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
    
    # Log the start of the backtest
    execution_id = None
    if logger:
        execution_id = logger.log_execution(
            execution_name=f"backtest_strategy_{strategy_params.strategy_id}",
            description=f"Backtest of strategy {strategy_params.strategy_id} on {ticker}",
            parameters=json.dumps(strategy_params.to_dict(), cls=DateTimeEncoder),
            script_id=None
        )
    
    try:
        # Determine date range if not provided
        if not start_date or not end_date:
            date_range_query = """
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM price_data_5min
            WHERE ticker = ?
            """
            min_date, max_date = db_manager.conn.execute(date_range_query, (ticker,)).fetchone()
            
            if not start_date:
                start_date = min_date.strftime('%Y-%m-%d') if isinstance(min_date, datetime) else min_date
            if not end_date:
                end_date = max_date.strftime('%Y-%m-%d') if isinstance(max_date, datetime) else max_date
                
        print(f"Backtesting strategy {strategy_params.strategy_id} for {ticker} from {start_date} to {end_date}")
        
        # Step 1: Create a daily signals table with trend indicators
        daily_signals_query = f"""
        WITH daily_data AS (
            SELECT
                ticker,
                date,
                open,
                high,
                low,
                close,
                volume,
                -- Simple Moving Averages with parameterized periods
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {strategy_params.sma_fast_period-1} PRECEDING AND CURRENT ROW) AS sma_fast,
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {strategy_params.sma_slow_period-1} PRECEDING AND CURRENT ROW) AS sma_slow,
                -- Volatility
                STDDEV(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {strategy_params.volatility_period-1} PRECEDING AND CURRENT ROW) / 
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {strategy_params.volatility_period-1} PRECEDING AND CURRENT ROW) AS volatility,
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
            sma_fast,
            sma_slow,
            volatility,
            daily_return,
            -- Trend signals
            CASE 
                WHEN close > sma_fast AND sma_fast > sma_slow THEN 'UPTREND'
                WHEN close < sma_fast AND sma_fast < sma_slow THEN 'DOWNTREND'
                ELSE 'NEUTRAL'
            END AS trend,
            -- Trend change signals
            CASE
                WHEN close > sma_fast AND LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) <= LAG(sma_fast, 1) OVER (PARTITION BY ticker ORDER BY date) THEN 'BUY_SIGNAL'
                WHEN close < sma_fast AND LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) >= LAG(sma_fast, 1) OVER (PARTITION BY ticker ORDER BY date) THEN 'SELL_SIGNAL'
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
        hourly_signals_query = f"""
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
                -- Calculate average gain and average loss over parameterized period
                AVG(CASE WHEN price_change > 0 THEN price_change ELSE 0 END) OVER (
                    PARTITION BY ticker ORDER BY date ROWS BETWEEN {strategy_params.rsi_period-1} PRECEDING AND CURRENT ROW
                ) AS avg_gain,
                AVG(CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END) OVER (
                    PARTITION BY ticker ORDER BY date ROWS BETWEEN {strategy_params.rsi_period-1} PRECEDING AND CURRENT ROW
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
        trading_signals_query = f"""
        WITH entry_signals AS (
            SELECT
                date,
                close,
                trend,
                daily_signal,
                rsi_14,
                hour,
                -- Generate entry signals based on trend and RSI with parameterized thresholds
                CASE
                    WHEN trend = 'UPTREND' AND rsi_14 < {strategy_params.rsi_buy_threshold} THEN 'BUY'
                    WHEN trend = 'DOWNTREND' AND rsi_14 > {strategy_params.rsi_sell_threshold} THEN 'SELL'
                    WHEN daily_signal = 'BUY_SIGNAL' AND rsi_14 < {strategy_params.rsi_buy_signal_threshold} THEN 'BUY'
                    WHEN daily_signal = 'SELL_SIGNAL' AND rsi_14 > {strategy_params.rsi_sell_signal_threshold} THEN 'SELL'
                    ELSE NULL
                END AS signal,
                -- Add lag signals for detecting changes
                LAG(
                    CASE
                        WHEN trend = 'UPTREND' AND rsi_14 < {strategy_params.rsi_buy_threshold} THEN 'BUY'
                        WHEN trend = 'DOWNTREND' AND rsi_14 > {strategy_params.rsi_sell_threshold} THEN 'SELL'
                        WHEN daily_signal = 'BUY_SIGNAL' AND rsi_14 < {strategy_params.rsi_buy_signal_threshold} THEN 'BUY'
                        WHEN daily_signal = 'SELL_SIGNAL' AND rsi_14 > {strategy_params.rsi_sell_signal_threshold} THEN 'SELL'
                        ELSE NULL
                    END, 1
                ) OVER (ORDER BY date) AS prev_signal,
                -- Row number within signal type to identify duplicates
                ROW_NUMBER() OVER (PARTITION BY 
                    CASE
                        WHEN trend = 'UPTREND' AND rsi_14 < {strategy_params.rsi_buy_threshold} THEN 'BUY'
                        WHEN trend = 'DOWNTREND' AND rsi_14 > {strategy_params.rsi_sell_threshold} THEN 'SELL'
                        WHEN daily_signal = 'BUY_SIGNAL' AND rsi_14 < {strategy_params.rsi_buy_signal_threshold} THEN 'BUY'
                        WHEN daily_signal = 'SELL_SIGNAL' AND rsi_14 > {strategy_params.rsi_sell_signal_threshold} THEN 'SELL'
                        ELSE NULL
                    END ORDER BY date) AS signal_row_num
            FROM temp_hourly_signals
        )
        SELECT
            date,
            close,
            trend,
            rsi_14,
            hour,
            signal,
            -- Only take trades when signal changes and is different from previous signal
            CASE
                WHEN signal = 'BUY' AND (prev_signal IS NULL OR prev_signal != 'BUY') AND signal_row_num = 1 THEN 'ENTER_LONG'
                WHEN signal = 'SELL' AND (prev_signal IS NULL OR prev_signal != 'SELL') AND signal_row_num = 1 THEN 'ENTER_SHORT'
                ELSE NULL
            END AS trade_action
        FROM entry_signals
        ORDER BY date
        """
        
        # Execute the trading signals query
        trading_signals = db_manager.query_to_df(trading_signals_query)
        
        # Step 4: Simulate the trades with improved position management and exit logic
        def simulate_trades(signals_df):
            signals_df = signals_df.copy()
            
            # Add columns for trade tracking
            signals_df['position'] = 0
            signals_df['entry_price'] = None
            signals_df['exit_price'] = None
            signals_df['trade_return'] = None
            signals_df['cumulative_return'] = 1.0
            
            position = 0
            entry_price = 0
            trade_returns = []
            
            # Use risk management parameters from strategy_params
            stop_loss_pct = strategy_params.stop_loss_pct
            take_profit_pct = strategy_params.take_profit_pct
            max_hold_periods = strategy_params.max_holding_periods
            periods_in_trade = 0
            
            # Pre-process duplicate signals
            last_action = None
            for i, row in signals_df.iterrows():
                # Skip duplicate trade signals
                if row['trade_action'] == last_action and row['trade_action'] is not None:
                    signals_df.at[i, 'trade_action'] = None
                else:
                    last_action = row['trade_action']
            
            # Main simulation loop
            for i, row in signals_df.iterrows():
                current_price = row['close']
                trade_action = row['trade_action']
                
                # Update periods in trade if in a position
                if position != 0:
                    periods_in_trade += 1
                
                # Check for exit conditions if in a position
                exit_reason = None
                
                if position == 1:  # Long position
                    # Check stop loss
                    if current_price < entry_price * (1 - stop_loss_pct):
                        exit_reason = "STOP_LOSS"
                    # Check take profit
                    elif current_price > entry_price * (1 + take_profit_pct):
                        exit_reason = "TAKE_PROFIT"
                    # Check time-based exit
                    elif periods_in_trade >= max_hold_periods:
                        exit_reason = "TIME_EXIT"
                    # Check for reversal signal
                    elif trade_action == 'ENTER_SHORT':
                        exit_reason = "SIGNAL_REVERSAL"
                
                elif position == -1:  # Short position
                    # Check stop loss
                    if current_price > entry_price * (1 + stop_loss_pct):
                        exit_reason = "STOP_LOSS"
                    # Check take profit
                    elif current_price < entry_price * (1 - take_profit_pct):
                        exit_reason = "TAKE_PROFIT"
                    # Check time-based exit
                    elif periods_in_trade >= max_hold_periods:
                        exit_reason = "TIME_EXIT"
                    # Check for reversal signal
                    elif trade_action == 'ENTER_LONG':
                        exit_reason = "SIGNAL_REVERSAL"
                
                # Process exit if needed
                if exit_reason and position != 0:
                    if position == 1:  # Exit long position
                        exit_price = current_price
                        trade_return = (exit_price / entry_price) - 1
                    else:  # Exit short position
                        exit_price = current_price
                        trade_return = 1 - (exit_price / entry_price)
                    
                    signals_df.at[i, 'exit_price'] = exit_price
                    signals_df.at[i, 'trade_return'] = trade_return
                    signals_df.at[i, 'trade_action'] = f"EXIT_{exit_reason}"
                    trade_returns.append(trade_return)
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    periods_in_trade = 0
                
                # Process entry if no position and we have an entry signal
                if position == 0 and (trade_action == 'ENTER_LONG' or trade_action == 'ENTER_SHORT'):
                    if trade_action == 'ENTER_LONG':
                        position = 1
                    else:  # ENTER_SHORT
                        position = -1
                    
                    entry_price = current_price
                    periods_in_trade = 0
                
                # Update position tracking
                signals_df.at[i, 'position'] = position
                if position != 0 and signals_df.at[i, 'entry_price'] is None:
                    signals_df.at[i, 'entry_price'] = entry_price
            
            # Close any open position at the end of the period
            if position != 0 and len(signals_df) > 0:
                last_idx = signals_df.index[-1]
                last_close = signals_df.loc[last_idx, 'close']
                
                if position == 1:
                    trade_return = (last_close / entry_price) - 1
                else:  # position == -1
                    trade_return = 1 - (last_close / entry_price)
                
                signals_df.at[last_idx, 'exit_price'] = last_close
                signals_df.at[last_idx, 'trade_return'] = trade_return
                signals_df.at[last_idx, 'trade_action'] = "EXIT_EOD"
                trade_returns.append(trade_return)
            
            # Calculate cumulative returns
            trade_returns_df = signals_df[signals_df['trade_return'].notnull()].copy()
            if not trade_returns_df.empty:
                trade_returns_df['cumulative_return'] = (1 + trade_returns_df['trade_return']).cumprod()
                
                # Update the main DataFrame
                for idx, row in trade_returns_df.iterrows():
                    signals_df.at[idx, 'cumulative_return'] = row['cumulative_return']
            
            # Fill forward cumulative returns and positions
            signals_df['cumulative_return'] = signals_df['cumulative_return'].ffill()
            
            # Fill positions between entry and exit
            for i in range(1, len(signals_df)):
                if signals_df.iloc[i]['position'] == 0 and signals_df.iloc[i-1]['position'] != 0:
                    # Only continue position if no exit
                    if not (str(signals_df.iloc[i]['trade_action'] or '').startswith('EXIT_')):
                        signals_df.iloc[i, signals_df.columns.get_loc('position')] = signals_df.iloc[i-1]['position']
            
            return signals_df, pd.Series(trade_returns)
        
        # Run the simulation
        results_df, trade_returns = simulate_trades(trading_signals)
        
        # Calculate performance metrics
        metrics = calculate_metrics(trade_returns)
        
        # Log the results to the database
        results_table_name = f"strategy_{strategy_params.strategy_id}_results"
        
        # Create the results table if it doesn't exist
        db_manager.conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {results_table_name} (
            date TIMESTAMP,
            close DOUBLE,
            trend VARCHAR,
            rsi_14 DOUBLE,
            hour BIGINT,
            signal VARCHAR,
            trade_action VARCHAR,
            position BIGINT,
            entry_price DOUBLE,
            exit_price DOUBLE,
            trade_return DOUBLE,
            cumulative_return DOUBLE
        )
        """)
        
        # Clear any existing results
        db_manager.conn.execute(f"DELETE FROM {results_table_name}")
        
        # Create a backtest results table specifically for the parameterized strategies
        db_manager.conn.execute("""
        CREATE TABLE IF NOT EXISTS parameterized_backtest_results (
            strategy_id INTEGER,
            date TIMESTAMP,
            close DOUBLE,
            trend VARCHAR,
            rsi_14 DOUBLE,
            hour BIGINT,
            signal VARCHAR,
            trade_action VARCHAR,
            position BIGINT,
            entry_price DOUBLE,
            exit_price DOUBLE,
            trade_return DOUBLE,
            cumulative_return DOUBLE,
            exit_reason VARCHAR
        )
        """)
        
        # Delete any existing results for this strategy
        db_manager.conn.execute(f"DELETE FROM parameterized_backtest_results WHERE strategy_id = {strategy_params.strategy_id}")
        
        # Insert the results with strategy_id
        for _, row in results_df.iterrows():
            # Extract exit reason from trade_action if present
            exit_reason = None
            if isinstance(row['trade_action'], str) and row['trade_action'].startswith('EXIT_'):
                exit_reason = row['trade_action'].replace('EXIT_', '')
            
            db_manager.conn.execute("""
            INSERT INTO parameterized_backtest_results (
                strategy_id, date, close, trend, rsi_14, hour, signal, trade_action,
                position, entry_price, exit_price, trade_return, cumulative_return, exit_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_params.strategy_id, row['date'], row['close'], row['trend'], row['rsi_14'], row['hour'],
                row['signal'], row['trade_action'], row['position'],
                row['entry_price'], row['exit_price'], row['trade_return'], row['cumulative_return'], exit_reason
            ))
        
        # Update the model in metadata with performance metrics
        if logger:
            # Get the model_id for this strategy
            model_id_result = db_manager.conn.execute(
                "SELECT model_id FROM metadata_models WHERE model_name = ?",
                (f"Strategy_{strategy_params.strategy_id}",)
            ).fetchone()
            
            if model_id_result:
                model_id = model_id_result[0]
                # Log the training metrics
                logger.log_model_training(
                    model_id=model_id,
                    training_time=time.time() - start_time,
                    performance_metrics=metrics,
                    status="trained"
                )
        
        # Generate and save the performance plot if requested
        if plot_returns and not results_df.empty:
            plot_file_path = f"notebooks/{ticker}_strategy_{strategy_params.strategy_id}.png"
            
            plt.figure(figsize=(12, 8))
            plt.plot(results_df['date'], results_df['cumulative_return'])
            plt.title(f"Strategy {strategy_params.strategy_id} Performance for {ticker}")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.grid(True)
            
            # Add strategy parameters as text
            params_text = (
                f"SMA: {strategy_params.sma_fast_period}/{strategy_params.sma_slow_period}, "
                f"RSI: {strategy_params.rsi_period}, "
                f"Buy: {strategy_params.rsi_buy_threshold}/{strategy_params.rsi_buy_signal_threshold}, "
                f"Sell: {strategy_params.rsi_sell_threshold}/{strategy_params.rsi_sell_signal_threshold}"
            )
            plt.figtext(0.5, 0.01, params_text, ha='center')
            
            # Add performance metrics as text
            metrics_text = (
                f"Win Rate: {metrics['win_rate']:.2f}%, "
                f"Trades: {metrics['total_trades']}, "
                f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                f"Ann. Return: {metrics['annualized_return']:.2f}%"
            )
            plt.figtext(0.5, 0.03, metrics_text, ha='center')
            
            plt.savefig(plot_file_path)
            plt.close()
            
            print(f"Results saved to database and plot saved to {plot_file_path}")
        
        # Log success
        if logger and execution_id:
            logger.log_execution_end(
                execution_id,
                status="success",
                result_summary=f"Backtest completed successfully for strategy {strategy_params.strategy_id}",
                result_metrics=json.dumps(metrics)
            )
        
        return results_df, metrics
    
    except Exception as e:
        # Log failure
        if logger and execution_id:
            logger.log_execution_end(
                execution_id,
                status="error",
                result_summary=f"Error in backtest: {str(e)}",
                result_metrics=json.dumps({"error": str(e)})
            )
        print(f"Error backtesting parameterized strategy: {str(e)}")
        raise

if __name__ == "__main__":
    # Test with a sample strategy
    from scripts.models.strategy_generator import StrategyParameters
    
    # Create a sample strategy
    sample_strategy = StrategyParameters(
        strategy_id=999,
        sma_fast_period=10,
        sma_slow_period=30,
        rsi_period=14,
        rsi_buy_threshold=30,
        rsi_sell_threshold=70,
        rsi_buy_signal_threshold=40,
        rsi_sell_signal_threshold=60,
        volatility_period=20
    )
    
    # Run the backtest with plot generation
    results_df, metrics = backtest_parameterized_strategy(
        sample_strategy,
        ticker='SPY',
        plot_returns=True
    )
    
    # Print the performance metrics
    print("\nStrategy Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}") 