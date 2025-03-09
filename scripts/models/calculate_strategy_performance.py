"""
Strategy Performance Calculator

This script calculates performance metrics for each strategy
based on the trades in the trade_log table and updates the
strategy_performance_summary table.
"""

import duckdb
import pandas as pd
import numpy as np
import json
import traceback
import sys
from datetime import datetime
import uuid

def calculate_strategy_performance():
    """Calculate performance metrics for each strategy and update the summary table"""
    execution_id = str(uuid.uuid4().hex)
    
    try:
        # Connect to the database directly
        con = duckdb.connect('db/backtest.ddb')
        print(f"Connected to database. Execution ID: {execution_id}")
        
        # First, check if the strategy_performance_summary table exists
        table_exists = con.execute("SELECT name FROM sqlite_master WHERE name='strategy_performance_summary'").fetchone() is not None
        
        if not table_exists:
            print("Creating strategy_performance_summary table...")
            con.execute("""
            CREATE TABLE strategy_performance_summary (
                strategy_id INTEGER PRIMARY KEY,
                total_trades INTEGER,
                win_rate DOUBLE,
                profit_factor DOUBLE,
                avg_return DOUBLE,
                max_drawdown DOUBLE,
                sharpe_ratio DOUBLE,
                annualized_return DOUBLE,
                parameters VARCHAR
            )
            """)
        
        # Get the list of all strategy_ids in trade_log
        strategy_ids = con.execute("""
        SELECT DISTINCT strategy_id 
        FROM trade_log 
        WHERE strategy_id IS NOT NULL
        """).fetchdf()
        
        print(f"Found {len(strategy_ids)} strategies with trades")
        
        # Calculate performance metrics for each strategy using SQL
        con.execute("""
        -- First, delete existing records to avoid duplicates
        DELETE FROM strategy_performance_summary;
        
        -- Insert with calculated metrics
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
        )
        WITH trade_metrics AS (
            SELECT 
                strategy_id,
                COUNT(*) as total_trades,
                ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) as win_rate,
                CASE 
                    WHEN SUM(CASE WHEN trade_return < 0 THEN ABS(trade_return) ELSE 0 END) = 0 THEN NULL
                    ELSE SUM(CASE WHEN trade_return > 0 THEN trade_return ELSE 0 END) / 
                         NULLIF(SUM(CASE WHEN trade_return < 0 THEN ABS(trade_return) ELSE 0 END), 0)
                END as profit_factor,
                AVG(trade_return) as avg_return
            FROM trade_log
            WHERE trade_return IS NOT NULL
            GROUP BY strategy_id
        ),
        strategy_params AS (
            SELECT 
                m.model_id as strategy_id,
                m.parameters
            FROM metadata_models m
            WHERE m.model_type = 'strategy'
        )
        SELECT 
            tm.strategy_id,
            tm.total_trades,
            tm.win_rate,
            tm.profit_factor,
            tm.avg_return,
            -- Placeholder for max_drawdown (requires time series calculation)
            0.0 as max_drawdown,
            -- Simple approximation for Sharpe ratio
            CASE 
                WHEN STDDEV(t.trade_return) = 0 THEN NULL
                ELSE AVG(t.trade_return) / NULLIF(STDDEV(t.trade_return), 0) * SQRT(252)
            END as sharpe_ratio,
            -- Annualized return (simplified)
            AVG(t.trade_return) * 252 as annualized_return,
            sp.parameters
        FROM trade_metrics tm
        JOIN trade_log t ON tm.strategy_id = t.strategy_id
        LEFT JOIN strategy_params sp ON tm.strategy_id = sp.strategy_id
        GROUP BY tm.strategy_id, tm.total_trades, tm.win_rate, tm.profit_factor, tm.avg_return, sp.parameters
        """)
        
        # Calculate the max drawdown separately (this is more complex and requires custom logic)
        strategies = con.execute("SELECT strategy_id FROM strategy_performance_summary").fetchdf()
        
        for _, row in strategies.iterrows():
            strategy_id = row['strategy_id']
            
            # Get trades for this strategy
            trades = con.execute(f"""
            SELECT 
                date,
                trade_return
            FROM trade_log
            WHERE strategy_id = {strategy_id} AND trade_return IS NOT NULL
            ORDER BY date
            """).fetchdf()
            
            # Calculate max drawdown if we have enough trades
            max_drawdown = 0.0
            if len(trades) > 1:
                # Convert returns to equity curve
                equity = (1 + trades['trade_return']).cumprod()
                running_max = equity.cummax()
                drawdown = (equity / running_max) - 1
                max_drawdown = abs(drawdown.min())
                
                # Update the record with the calculated max drawdown
                con.execute(f"""
                UPDATE strategy_performance_summary
                SET max_drawdown = {max_drawdown}
                WHERE strategy_id = {strategy_id}
                """)
        
        # Get the results
        results = con.execute("""
        SELECT 
            strategy_id,
            total_trades,
            win_rate,
            profit_factor,
            avg_return,
            max_drawdown,
            sharpe_ratio,
            annualized_return
        FROM strategy_performance_summary
        ORDER BY sharpe_ratio DESC NULLS LAST
        LIMIT 10
        """).fetchdf()
        
        print("\nTop 10 strategies by Sharpe ratio:")
        print(results)
        
        # Count the updated records
        count = con.execute("SELECT COUNT(*) FROM strategy_performance_summary").fetchone()[0]
        print(f"\nUpdated performance metrics for {count} strategies")
        
        # Log the execution in the metadata tables directly
        try:
            # Log execution
            con.execute(f"""
            INSERT INTO metadata_executions (
                execution_id, 
                execution_name, 
                description, 
                start_time, 
                end_time, 
                status, 
                result_summary
            ) VALUES (
                '{execution_id}',
                'calculate_strategy_performance',
                'Calculate performance metrics for all strategies',
                '{datetime.now().isoformat()}',
                '{datetime.now().isoformat()}',
                'COMPLETED',
                'Updated performance metrics for {count} strategies'
            )
            """)
            
            # Log query
            query_id = str(uuid.uuid4().hex)
            con.execute(f"""
            INSERT INTO metadata_queries (
                query_id,
                query_name,
                query_type,
                sql_text,
                execution_time,
                execution_id,
                status
            ) VALUES (
                '{query_id}',
                'calculate_strategy_performance',
                'ANALYTICS',
                'See detailed implementation in calculate_strategy_performance.py',
                0,
                '{execution_id}',
                'COMPLETED'
            )
            """)
            
            con.commit()
            print("Logged execution in metadata tables")
        except Exception as e:
            print(f"Warning: Could not log to metadata tables: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"Error calculating strategy performance: {str(e)}")
        traceback.print_exc()
        return False
    finally:
        if 'con' in locals():
            con.close()

if __name__ == "__main__":
    calculate_strategy_performance() 