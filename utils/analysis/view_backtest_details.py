import duckdb
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import traceback
from pathlib import Path

# Get the database path
ROOT_DIR = Path(__file__).parent.parent.parent
DB_PATH = ROOT_DIR / "db" / "backtest.ddb"

def analyze_backtest_details():
    """
    Analyze and display detailed information about the backtest trades with surrounding context.
    """
    try:
        # Connect directly to the database
        conn = duckdb.connect(str(DB_PATH))
        
        # Get the dates around which trades occurred
        trade_dates = conn.execute('''
            SELECT date 
            FROM multi_timeframe_backtest_results 
            WHERE trade_action IS NOT NULL 
            ORDER BY date
        ''').fetchall()

        if not trade_dates:
            print("No trades found in the backtest results.")
            return

        # For each trade, show the data leading up to and following the trade
        window_days = 2  # Show 2 days before and after each trade

        for trade_date in trade_dates:
            trade_date = trade_date[0] if trade_date and len(trade_date) > 0 else None  # Extract date from tuple
            if not trade_date:
                continue
                
            # Calculate window bounds
            start_date = trade_date - timedelta(days=window_days)
            end_date = trade_date + timedelta(days=window_days)
            
            # Query data for the window
            results = conn.execute('''
                SELECT 
                    date,
                    close,
                    trend,
                    rsi_14,
                    signal,
                    trade_action,
                    position,
                    entry_price,
                    exit_price,
                    trade_return,
                    cumulative_return
                FROM multi_timeframe_backtest_results 
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            ''', (start_date, end_date)).fetchall()
            
            if not results:
                print(f"No data found for date range: {start_date} to {end_date}")
                continue
                
            # Convert to DataFrame for better display
            columns = [
                'Date', 'Close', 'Trend', 'RSI', 'Signal', 'Trade Action', 
                'Position', 'Entry Price', 'Exit Price', 'Trade Return', 'Cumulative Return'
            ]
            df = pd.DataFrame(results, columns=columns)
            
            # Display the results
            print("\n" + "=" * 100)
            print(f"Trade Analysis Window: {start_date} to {end_date}")
            print("=" * 100)
            
            # Highlight the actual trade row
            for i, row in df.iterrows():
                is_trade_row = row['Date'] == trade_date
                prefix = ">>> " if is_trade_row else "    "
                
                # Format the row for display
                date_str = str(row['Date'])
                close = f"${row['Close']:.2f}" if row['Close'] is not None else "N/A"
                trend = row['Trend'] if row['Trend'] is not None else "N/A"
                rsi = f"{row['RSI']:.2f}" if row['RSI'] is not None else "N/A"
                signal = row['Signal'] if row['Signal'] is not None else "N/A"
                action = row['Trade Action'] if row['Trade Action'] is not None else "-"
                position = str(row['Position']) if row['Position'] is not None else "0"
                entry = f"${row['Entry Price']:.2f}" if row['Entry Price'] is not None else "N/A"
                exit = f"${row['Exit Price']:.2f}" if row['Exit Price'] is not None else "N/A"
                ret = f"{row['Trade Return']*100:.2f}%" if row['Trade Return'] is not None else "N/A"
                cum_ret = f"{row['Cumulative Return']*100:.2f}%" if row['Cumulative Return'] is not None else "N/A"
                
                print(f"{prefix}{date_str} | Close: {close} | Trend: {trend} | RSI: {rsi} | Signal: {signal}")
                if is_trade_row:
                    print(f"{prefix}Action: {action} | Position: {position} | Entry: {entry} | Exit: {exit} | Return: {ret} | Cum Return: {cum_ret}")
                    print("-" * 100)
        
        print("\nOverall Backtest Performance")
        print("=" * 50)

        # Get overall performance metrics
        results = conn.execute('''
            SELECT 
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(CASE WHEN trade_action IS NOT NULL THEN 1 END) as total_trades,
                COUNT(CASE WHEN trade_return > 0 THEN 1 END) as winning_trades,
                MAX(cumulative_return) as max_return
            FROM multi_timeframe_backtest_results
        ''').fetchone()

        if not results or len(results) < 5:
            print("No performance metrics available.")
            return
            
        start_date, end_date, total_trades, winning_trades, max_return = results

        if not all([start_date, end_date, max_return is not None]):
            print("Incomplete performance metrics available.")
            return
            
        days_diff = (end_date - start_date).days
        years_diff = days_diff / 365.0
        annualized_return = ((1 + max_return) ** (1 / years_diff)) - 1 if max_return is not None and years_diff > 0 else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        print(f"Backtest Period: {start_date} to {end_date} ({days_diff} days)")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Return: {max_return*100:.2f}%")
        print(f"Annualized Return: {annualized_return*100:.2f}%")
    
    except Exception as e:
        print(f"Error analyzing backtest details: {str(e)}")
        traceback.print_exc()
    finally:
        # Close the connection
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    analyze_backtest_details() 