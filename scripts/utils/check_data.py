import duckdb
import pandas as pd

def check_database():
    # Connect to the database
    con = duckdb.connect('db/backtest.ddb')
    
    # Check available tables
    tables = con.execute('SELECT table_name FROM information_schema.tables').fetchall()
    print("Available tables:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Check columns in price data tables
    price_tables = [table[0] for table in tables if table[0].startswith('price_data')]
    for table in price_tables:
        try:
            columns = con.execute(f'SELECT column_name FROM information_schema.columns WHERE table_name = \'{table}\'').fetchall()
            print(f"\n{table} columns:")
            for col in columns:
                print(f"  - {col[0]}")
            
            # Check data ranges
            result = con.execute(f'SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as count FROM {table}').fetchone()
            print(f"  Date Range: {result[0]} to {result[1]}, Total Rows: {result[2]}")
            
            # Get sample data
            sample = con.execute(f'SELECT * FROM {table} LIMIT 1').fetchone()
            print(f"  Sample row: {sample}")
        except Exception as e:
            print(f"Error querying {table}: {e}")
    
    # Check if multi_timeframe_backtest_results exists and has data
    if ('multi_timeframe_backtest_results',) in tables:
        count = con.execute('SELECT COUNT(*) FROM multi_timeframe_backtest_results').fetchone()[0]
        print(f"\nmulti_timeframe_backtest_results has {count} rows")
        
        # Check trade actions
        trade_actions = con.execute('SELECT trade_action, COUNT(*) FROM multi_timeframe_backtest_results WHERE trade_action IS NOT NULL GROUP BY trade_action').fetchall()
        print("Trade actions:")
        for action in trade_actions:
            print(f"  {action[0]}: {action[1]} occurrences")
    
    # Check if all_price_data view exists
    if ('all_price_data',) in tables:
        try:
            # Check structure
            columns = con.execute('SELECT column_name FROM information_schema.columns WHERE table_name = \'all_price_data\'').fetchall()
            print("\nall_price_data columns:")
            for col in columns:
                print(f"  - {col[0]}")
            
            # Check data by timeframe and ticker
            result = con.execute('SELECT timeframe, MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as count FROM all_price_data GROUP BY timeframe ORDER BY CASE timeframe WHEN \'5min\' THEN 1 WHEN \'15min\' THEN 2 WHEN \'30min\' THEN 3 WHEN \'60min\' THEN 4 WHEN \'1d\' THEN 5 WHEN \'1w\' THEN 6 ELSE 7 END').fetchall()
            print("\nall_price_data by timeframe:")
            for row in result:
                print(f"  Timeframe: {row[0]}, Date Range: {row[1]} to {row[2]}, Count: {row[3]}")
        except Exception as e:
            print(f"Error querying all_price_data: {e}")
    
    # Close the connection
    con.close()

if __name__ == "__main__":
    check_database() 