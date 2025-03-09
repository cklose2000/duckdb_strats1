"""
Backtest Analysis Script

This script analyzes the results of the backtest run,
providing detailed information about strategies and trades.
"""

import duckdb
import pandas as pd
import json
import traceback
import sys

def analyze_backtest():
    # Connect to the database
    try:
        con = duckdb.connect('db/backtest.ddb')
        print(f"Successfully connected to database 'db/backtest.ddb'")
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {str(e)}")
        traceback.print_exc()
        return
    
    # 1. Check overall trade statistics
    print("=" * 80)
    print("TRADE STATISTICS")
    print("=" * 80)
    
    try:
        # Debug: check table existence
        tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchdf()
        print(f"Database contains {len(tables)} tables")
        
        if 'trade_log' not in tables['name'].values:
            print("ERROR: trade_log table does not exist!")
            print("Available tables:")
            print(tables)
            return
            
        # Count total trades and strategies
        trade_stats = con.execute("""
        SELECT 
            COUNT(*) as total_trades, 
            COUNT(DISTINCT strategy_id) as total_strategies,
            COUNT(DISTINCT execution_id) as total_executions
        FROM trade_log
        """).fetchone()
        
        print(f"Total trades: {trade_stats[0]}")
        print(f"Total strategies: {trade_stats[1]}")
        print(f"Total executions: {trade_stats[2]}")
        
        if trade_stats[0] == 0:
            print("WARNING: No trades found in trade_log table!")
            # Debug: Show table schema
            schema = con.execute("PRAGMA table_info(trade_log)").fetchdf()
            print("Trade log schema:")
            print(schema)
            return
        
        # Trades by direction
        direction_stats = con.execute("""
        SELECT 
            direction, 
            COUNT(*) as count,
            ROUND(AVG(CASE WHEN trade_return IS NOT NULL THEN trade_return ELSE 0 END) * 100, 2) as avg_return_pct,
            ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) as win_rate
        FROM trade_log
        GROUP BY direction
        """).fetchall()
        
        print("\nTrades by direction:")
        for direction in direction_stats:
            print(f"  {direction[0]}: {direction[1]} trades, {direction[2]}% avg return, {direction[3]}% win rate")
        
        # Exit reasons
        exit_reasons = con.execute("""
        SELECT 
            exit_reason, 
            COUNT(*) as count,
            ROUND(AVG(CASE WHEN trade_return IS NOT NULL THEN trade_return ELSE 0 END) * 100, 2) as avg_return_pct
        FROM trade_log
        WHERE exit_reason IS NOT NULL
        GROUP BY exit_reason
        ORDER BY count DESC
        """).fetchall()
        
        print("\nTrades by exit reason:")
        for reason in exit_reasons:
            print(f"  {reason[0]}: {reason[1]} trades, {reason[2]}% avg return")
    except Exception as e:
        print(f"ERROR analyzing trade statistics: {str(e)}")
        traceback.print_exc()
    
    # 2. Top strategies by Sharpe ratio
    print("\n" + "=" * 80)
    print("TOP STRATEGIES BY SHARPE RATIO")
    print("=" * 80)
    
    try:
        # Debug: check table existence
        tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchdf()
        if 'strategy_performance_summary' not in tables['name'].values:
            print("ERROR: strategy_performance_summary table does not exist!")
            return
            
        # Check if table has any records
        count = con.execute("SELECT COUNT(*) FROM strategy_performance_summary").fetchone()[0]
        print(f"Found {count} records in strategy_performance_summary")
        
        if count == 0:
            print("WARNING: No strategy performance records found!")
            return
            
        top_strategies = con.execute("""
        SELECT 
            s.strategy_id,
            s.total_trades,
            s.win_rate,
            s.profit_factor,
            s.sharpe_ratio,
            s.annualized_return,
            s.parameters
        FROM strategy_performance_summary s
        WHERE s.total_trades >= 2 AND s.sharpe_ratio IS NOT NULL
        ORDER BY s.sharpe_ratio DESC
        LIMIT 10
        """).fetchall()
        
        if top_strategies:
            for i, strat in enumerate(top_strategies):
                print(f"\n{i+1}. Strategy {strat[0]}")
                print(f"   Trades: {strat[1]}, Win Rate: {strat[2]}%, Sharpe: {strat[4]:.2f}")
                print(f"   Profit Factor: {strat[3]}, Ann. Return: {strat[5]:.2f}%")
                
                # Parse parameters
                try:
                    params = json.loads(strat[6])
                    print(f"   SMA: {params['sma_fast_period']}/{params['sma_slow_period']}, RSI: {params['rsi_period']}")
                    print(f"   RSI Buy/Sell: {params['rsi_buy_threshold']}/{params['rsi_sell_threshold']}")
                    print(f"   Stop Loss: {params.get('stop_loss_pct', 0.02)*100}%, Take Profit: {params.get('take_profit_pct', 0.05)*100}%")
                except Exception as e:
                    print(f"   Error parsing parameters: {str(e)}")
                    print(f"   Raw parameters: {strat[6]}")
                
                # Show trades for this strategy
                trades = con.execute(f"""
                SELECT 
                    date, 
                    action, 
                    price, 
                    entry_price, 
                    exit_price, 
                    trade_return, 
                    exit_reason
                FROM trade_log
                WHERE strategy_id = {strat[0]} AND trade_return IS NOT NULL
                ORDER BY date
                """).fetchall()
                
                print("\n   Trades:")
                for trade in trades:
                    date, action, price, entry, exit, ret, reason = trade
                    ret_pct = f"{ret*100:.2f}%" if ret is not None else "N/A"
                    print(f"     {date} - {action} - Entry: {entry}, Exit: {exit}, Return: {ret_pct}, Reason: {reason}")
        else:
            print("No strategies with valid Sharpe ratio found")
            
            # Debug: show all strategies regardless of criteria
            print("\nDebugging: All strategies in summary table:")
            all_strats = con.execute("""
            SELECT 
                strategy_id, 
                total_trades, 
                win_rate, 
                profit_factor, 
                sharpe_ratio
            FROM strategy_performance_summary
            LIMIT 10
            """).fetchdf()
            print(all_strats)
    except Exception as e:
        print(f"ERROR analyzing top strategies: {str(e)}")
        traceback.print_exc()
    
    # 3. System metadata
    print("\n" + "=" * 80)
    print("METADATA SUMMARY")
    print("=" * 80)
    
    try:
        # Count all metadata objects
        metadata_counts = con.execute("""
        SELECT 
            (SELECT COUNT(*) FROM metadata_scripts) as scripts,
            (SELECT COUNT(*) FROM metadata_queries) as queries,
            (SELECT COUNT(*) FROM metadata_models) as models,
            (SELECT COUNT(*) FROM metadata_executions) as executions,
            (SELECT COUNT(*) FROM metadata_datasets) as datasets
        """).fetchone()
        
        print("Metadata object counts:")
        print(f"  Scripts: {metadata_counts[0]}")
        print(f"  Queries: {metadata_counts[1]}")
        print(f"  Models: {metadata_counts[2]}")
        print(f"  Executions: {metadata_counts[3]}")
        print(f"  Datasets: {metadata_counts[4]}")
        
        # Show last execution
        last_exec = con.execute("""
        SELECT 
            execution_id, 
            execution_name,
            description,
            start_time,
            end_time,
            status,
            result_summary
        FROM metadata_executions
        ORDER BY start_time DESC
        LIMIT 1
        """).fetchone()
        
        if last_exec:
            print("\nLast execution:")
            print(f"  ID: {last_exec[0]}")
            print(f"  Name: {last_exec[1]}")
            print(f"  Description: {last_exec[2]}")
            print(f"  Start: {last_exec[3]}")
            print(f"  End: {last_exec[4]}")
            print(f"  Status: {last_exec[5]}")
            print(f"  Summary: {last_exec[6]}")
    except Exception as e:
        print(f"ERROR analyzing metadata: {str(e)}")
        traceback.print_exc()
    
    # Close the connection
    con.close()

if __name__ == "__main__":
    try:
        analyze_backtest() 
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 