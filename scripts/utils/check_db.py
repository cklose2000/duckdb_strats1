import duckdb
import pandas as pd

def check_database():
    try:
        # Connect to database
        con = duckdb.connect('db/backtest.ddb')
        print("Successfully connected to database")
        
        # List all tables
        tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchdf()
        print(f"\nDatabase contains {len(tables)} tables")
        print(tables.head(20))
        
        # Check for strategy_performance_summary table
        has_summary = con.execute("SELECT name FROM sqlite_master WHERE name='strategy_performance_summary'").fetchone() is not None
        print(f"\nstrategy_performance_summary table exists: {has_summary}")
        
        if has_summary:
            # Check schema
            schema = con.execute("PRAGMA table_info(strategy_performance_summary)").fetchdf()
            print("\nSchema for strategy_performance_summary:")
            print(schema)
            
            # Check row count
            count = con.execute("SELECT COUNT(*) FROM strategy_performance_summary").fetchone()[0]
            print(f"\nRows in strategy_performance_summary: {count}")
            
            if count > 0:
                # Sample data
                sample = con.execute("SELECT * FROM strategy_performance_summary LIMIT 5").fetchdf()
                print("\nSample data from strategy_performance_summary:")
                print(sample)
        
        # Check for trade_log table
        has_trades = con.execute("SELECT name FROM sqlite_master WHERE name='trade_log'").fetchone() is not None
        print(f"\ntrade_log table exists: {has_trades}")
        
        if has_trades:
            # Check schema
            schema = con.execute("PRAGMA table_info(trade_log)").fetchdf()
            print("\nSchema for trade_log:")
            print(schema)
            
            # Count trades
            count = con.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]
            print(f"\nTrades in trade_log: {count}")
            
            if count > 0:
                # Sample data
                sample = con.execute("SELECT * FROM trade_log LIMIT 5").fetchdf()
                print("\nSample data from trade_log:")
                print(sample)
                
                # Trade statistics
                stats = con.execute("""
                SELECT 
                    direction, 
                    COUNT(*) as count, 
                    AVG(trade_return) as avg_return,
                    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END)::FLOAT/COUNT(*) as win_rate
                FROM trade_log
                GROUP BY direction
                """).fetchdf()
                print("\nTrade statistics by direction:")
                print(stats)
        
        # Check metadata tables
        for table in ['metadata_scripts', 'metadata_queries', 'metadata_models', 'metadata_executions']:
            has_table = con.execute(f"SELECT name FROM sqlite_master WHERE name='{table}'").fetchone() is not None
            print(f"\n{table} exists: {has_table}")
            
            if has_table:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"Rows in {table}: {count}")
                
                if count > 0:
                    sample = con.execute(f"SELECT * FROM {table} LIMIT 2").fetchdf()
                    print(f"Sample data from {table}:")
                    print(sample)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'con' in locals():
            con.close()

if __name__ == "__main__":
    check_database() 