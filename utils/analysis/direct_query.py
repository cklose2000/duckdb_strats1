"""
Direct database query utility for analyzing backtest results.

This script bypasses the DBManager to query the database directly, which can be useful 
for troubleshooting database access issues or when simpler access is needed.
"""

import duckdb
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Get the database path
ROOT_DIR = Path(__file__).parent.parent.parent
DB_PATH = ROOT_DIR / "db" / "backtest.ddb"

def view_trades_direct():
    """
    View and analyze the trades executed during backtest using a direct DuckDB connection.
    """
    try:
        # Connect directly to the database
        conn = duckdb.connect(str(DB_PATH))
        
        # Query the trades
        results = conn.execute('''
            SELECT 
                date, 
                trade_action, 
                position, 
                entry_price, 
                exit_price, 
                trade_return 
            FROM multi_timeframe_backtest_results 
            WHERE trade_action IS NOT NULL 
            ORDER BY date
        ''').fetchall()

        # Display the results
        print("\nTrade Summary (Direct Query):")
        print("=" * 80)
        print(f"{'Date':<20} {'Action':<15} {'Position':<10} {'Entry Price':<15} {'Exit Price':<15} {'Return %':<10}")
        print("-" * 80)

        if not results:
            print("No trades found in the backtest results.")
            return

        for row in results:
            date = str(row[0])
            action = row[1] if row[1] is not None else "N/A"
            position = row[2] if row[2] is not None else 0
            entry_price = row[3] if row[3] is not None else 0
            exit_price = row[4] if row[4] is not None else 0
            trade_return = row[5] if row[5] is not None else 0
            
            print(f"{date:<20} {action:<15} {position:<10} ${entry_price:<14.2f} ${exit_price:<14.2f} {trade_return*100:<9.2f}%")

        print("=" * 80)

        # Show overall statistics
        total_trades = len(results)
        winning_trades = sum(1 for row in results if row[5] is not None and row[5] > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        
        # Get overall performance
        results = conn.execute('''
            SELECT 
                MIN(date) as start_date,
                MAX(date) as end_date,
                MAX(cumulative_return) as max_return
            FROM multi_timeframe_backtest_results
        ''').fetchone()
        
        if results and all(r is not None for r in results):
            start_date, end_date, max_return = results
            days_diff = (end_date - start_date).days
            years_diff = days_diff / 365.0
            annualized_return = ((1 + max_return) ** (1 / years_diff)) - 1 if years_diff > 0 else 0
            
            print(f"Backtest Period: {start_date} to {end_date} ({days_diff} days)")
            print(f"Total Return: {max_return*100:.2f}%")
            print(f"Annualized Return: {annualized_return*100:.2f}%")
        
    except Exception as e:
        print(f"Error retrieving trade data: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the connection
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    view_trades_direct() 