import duckdb
import sys
import os
from pathlib import Path

# Get the database path
ROOT_DIR = Path(__file__).parent.parent.parent
DB_PATH = ROOT_DIR / "db" / "backtest.ddb"

def view_trades():
    """
    View and analyze the trades executed during backtest.
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
        print("\nTrade Summary:")
        print("=" * 80)
        print(f"{'Date':<20} {'Action':<15} {'Position':<10} {'Entry Price':<15} {'Exit Price':<15} {'Return %':<10}")
        print("-" * 80)

        if not results:
            print("No trades found in the backtest results.")
            return

        for row in results:
            if len(row) < 6:
                print(f"Warning: Unexpected data format in row: {row}")
                continue
                
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
        winning_trades = sum(1 for row in results if len(row) > 5 and row[5] is not None and row[5] > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")

    except Exception as e:
        print(f"Error retrieving trade data: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the connection
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    view_trades() 