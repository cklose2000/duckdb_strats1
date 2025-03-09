"""
Taapi.io Data Ingestion Script

This script downloads cryptocurrency and stock data from taapi.io API for different timeframes
and ingests it into the DuckDB database for backtesting.
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import from utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger
from utils.taapi_client import get_taapi_client

# Asset definitions
CRYPTO_ASSETS = [
    "BTC/USDT"  # Change from BTC/USD to BTC/USDT for Binance compatibility
]

STOCK_ASSETS = [
    "SPY",
    "QQQ"
]

# Timeframe definitions
TIMEFRAMES = [
    "5m",   # 5 minute
    "15m",  # 15 minute
    "30m",  # 30 minute
    "1h",   # 60 minute (1 hour)
    "4h",   # 240 minute (4 hour)
    "1d",   # daily
    "1w"    # weekly
]

# Map taapi.io timeframes to our database naming convention
TIMEFRAME_MAP = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
    "4h": "240min",
    "1d": "daily",
    "1w": "weekly"
}

def ingest_asset_data(
    asset, 
    asset_type, 
    timeframes=None, 
    limit=1000, 
    db_manager=None, 
    logger=None,
    taapi_client=None
):
    """
    Download and ingest data for a specific asset across multiple timeframes.
    
    Args:
        asset (str): Asset symbol (e.g., 'BTC/USD', 'SPY')
        asset_type (str): Type of asset ('crypto' or 'stocks')
        timeframes (list, optional): List of timeframes to download. Defaults to TIMEFRAMES.
        limit (int, optional): Maximum number of candles to download per timeframe. Defaults to 1000.
        db_manager (DBManager, optional): Database manager instance.
        logger (MetadataLogger, optional): Metadata logger instance.
        taapi_client (TaapiClient, optional): Taapi.io client instance.
        
    Returns:
        dict: Dictionary mapping timeframes to dataset IDs.
    """
    # Create necessary instances if not provided
    if db_manager is None:
        db_manager = DBManager()
        db_manager.connect()
        
    if logger is None:
        logger = MetadataLogger(db_manager.conn)
        
    if taapi_client is None:
        taapi_client = get_taapi_client()
        
    if timeframes is None:
        timeframes = TIMEFRAMES
        
    # Log script execution
    script_id = logger.log_script(
        __file__,
        description=f"Ingest {asset_type} data for {asset} from taapi.io API",
        parameters={
            "asset": asset,
            "asset_type": asset_type,
            "timeframes": timeframes,
            "limit": limit
        },
        script_type="data_ingestion"
    )
    
    start_time = time.time()
    
    # Dictionary to store dataset IDs
    dataset_ids = {}
    
    try:
        # Process each timeframe
        for timeframe in timeframes:
            print(f"Downloading {asset} data for {timeframe} timeframe...")
            
            # Standardize asset name for database
            if asset_type == 'crypto':
                ticker = asset.replace('/', '_')
            else:
                ticker = asset
                
            # Map timeframe to database naming convention
            db_timeframe = TIMEFRAME_MAP.get(timeframe, timeframe)
            
            # Table name based on timeframe
            table_name = f"price_data_{db_timeframe}"
            
            try:
                # Get price history from taapi.io
                df = taapi_client.get_price_history(
                    symbol=asset,
                    interval=timeframe,
                    limit=limit,
                    # Additional parameters for stocks
                    **({'type': 'stocks'} if asset_type == 'stocks' else {})
                )
                
                if df.empty:
                    print(f"No data received for {asset} on {timeframe}")
                    continue
                    
                print(f"  Received {len(df)} candles for {asset} on {timeframe}")
                
                # Drop the table if it exists
                db_manager.conn.execute(f"DROP TABLE IF EXISTS temp_{table_name}_{ticker}")
                
                # Register the DataFrame as a temporary table
                db_manager.conn.register(f"temp_{table_name}_{ticker}", df)
                
                # Create the main table if it doesn't exist
                if not db_manager.table_exists(table_name):
                    create_table_query = f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        date TIMESTAMP,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        volume DOUBLE,
                        ticker VARCHAR,
                        timeframe VARCHAR
                    );
                    """
                    db_manager.execute(create_table_query)
                
                # Insert data into the main table
                insert_query = f"""
                INSERT INTO {table_name} (date, open, high, low, close, volume, ticker, timeframe)
                SELECT 
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    ticker,
                    timeframe
                FROM temp_{table_name}_{ticker}
                ON CONFLICT (date, ticker, timeframe) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume;
                """
                
                try:
                    db_manager.execute(insert_query)
                except Exception as e:
                    # If conflict clause fails (first time), try without it
                    if "ON CONFLICT" in str(e):
                        insert_query = f"""
                        INSERT INTO {table_name} (date, open, high, low, close, volume, ticker, timeframe)
                        SELECT 
                            date,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            ticker,
                            timeframe
                        FROM temp_{table_name}_{ticker};
                        """
                        db_manager.execute(insert_query)
                    else:
                        raise
                
                # Clean up temporary table
                db_manager.conn.execute(f"DROP TABLE IF EXISTS temp_{table_name}_{ticker}")
                
                # Get row count and time range
                count_query = f"""
                SELECT COUNT(*) FROM {table_name} 
                WHERE ticker = '{ticker}' AND timeframe = '{db_timeframe}'
                """
                row_count = db_manager.conn.execute(count_query).fetchone()[0]
                
                time_range_query = f"""
                SELECT 
                    MIN(date) as min_date,
                    MAX(date) as max_date
                FROM {table_name}
                WHERE ticker = '{ticker}' AND timeframe = '{db_timeframe}'
                """
                time_range_result = db_manager.conn.execute(time_range_query).fetchone()
                time_range = {
                    "min_date": str(time_range_result[0]),
                    "max_date": str(time_range_result[1])
                }
                
                # Log the dataset
                dataset_id = logger.log_dataset(
                    dataset_name=f"{table_name}_{ticker}",
                    source_type="taapi.io API",
                    description=f"{asset} {timeframe} price data",
                    source_location=f"taapi.io API - {asset_type}/{asset}/{timeframe}",
                    schema={
                        "date": "TIMESTAMP",
                        "open": "DOUBLE",
                        "high": "DOUBLE",
                        "low": "DOUBLE",
                        "close": "DOUBLE",
                        "volume": "DOUBLE",
                        "ticker": "VARCHAR",
                        "timeframe": "VARCHAR"
                    },
                    row_count=row_count,
                    time_range=time_range,
                    script_id=script_id
                )
                
                # Store the dataset ID
                dataset_ids[timeframe] = dataset_id
                
                print(f"  Loaded {row_count} rows for {asset} {timeframe}")
                
            except Exception as e:
                print(f"Error processing {asset} {timeframe}: {e}")
        
        # Create a unified view that combines all timeframes if it doesn't exist
        try:
            view_exists = False
            try:
                # Try to check if the view exists using information_schema
                view_query = "SELECT table_name FROM information_schema.views WHERE table_name = 'all_price_data'"
                view_result = db_manager.conn.execute(view_query).fetchone()
                view_exists = view_result is not None
            except Exception:
                # If information_schema.views doesn't work, try another method
                try:
                    # Check if we can select from the view (if it exists)
                    db_manager.conn.execute("SELECT 1 FROM all_price_data LIMIT 1")
                    view_exists = True
                except Exception:
                    view_exists = False
            
            if not view_exists:
                # Check which tables actually exist
                table_exists_query = """
                SELECT table_name FROM information_schema.tables 
                WHERE table_name LIKE 'price_data_%'
                """
                existing_tables = db_manager.conn.execute(table_exists_query).fetchall()
                existing_tables = [t[0] for t in existing_tables]
                
                if len(existing_tables) > 0:
                    # Build the UNION query dynamically based on existing tables
                    union_parts = []
                    for table in existing_tables:
                        union_parts.append(f"SELECT * FROM {table}")
                    
                    union_query = " UNION ALL ".join(union_parts)
                    
                    create_unified_view_query = f"""
                    CREATE OR REPLACE VIEW all_price_data AS
                    {union_query};
                    """
                    
                    # Execute the view creation
                    db_manager.execute(create_unified_view_query)
                    print("Created unified view across all timeframes")
        except Exception as e:
            print(f"Warning: Could not create or verify unified view: {e}")
        
        # Log execution success
        execution_time = time.time() - start_time
        logger.log_script_execution(script_id, execution_time, "success")
        
        return dataset_ids
        
    except Exception as e:
        # Log execution failure
        execution_time = time.time() - start_time
        logger.log_script_execution(script_id, execution_time, f"failed: {str(e)}")
        raise

def main():
    # Create DB manager and logger
    db_manager = DBManager()
    db_manager.connect()
    logger = MetadataLogger(db_manager.conn)
    taapi_client = get_taapi_client()
    
    # Log execution start
    execution_id = logger.log_execution(
        execution_name="taapi_data_ingestion",
        description="Ingest price data from taapi.io for multiple assets and timeframes",
        parameters={
            "crypto_assets": CRYPTO_ASSETS,
            "stock_assets": STOCK_ASSETS,
            "timeframes": TIMEFRAMES
        }
    )
    
    asset_datasets = {}
    
    try:
        # Process each crypto asset
        for asset in CRYPTO_ASSETS:
            print(f"\nProcessing crypto asset: {asset}")
            datasets = ingest_asset_data(
                asset=asset,
                asset_type="crypto",
                db_manager=db_manager,
                logger=logger,
                taapi_client=taapi_client
            )
            asset_datasets[asset] = datasets
        
        # Process each stock asset
        for asset in STOCK_ASSETS:
            print(f"\nProcessing stock asset: {asset}")
            datasets = ingest_asset_data(
                asset=asset,
                asset_type="stocks",
                db_manager=db_manager,
                logger=logger,
                taapi_client=taapi_client
            )
            asset_datasets[asset] = datasets
        
        # Get a summary of all data - without relying on the all_price_data view
        table_query = """
        SELECT table_name FROM information_schema.tables 
        WHERE table_name LIKE 'price_data_%'
        """
        tables = db_manager.conn.execute(table_query).fetchall()
        tables = [t[0] for t in tables]
        
        summary_rows = []
        total_rows = 0
        
        for table in tables:
            try:
                # Get info for each price data table
                table_summary = db_manager.conn.execute(f"""
                SELECT 
                    ticker,
                    timeframe,
                    COUNT(*) as count,
                    MIN(date) as start_date,
                    MAX(date) as end_date
                FROM {table}
                GROUP BY ticker, timeframe
                """).fetchall()
                
                for row in table_summary:
                    summary_rows.append({
                        'ticker': row[0],
                        'timeframe': row[1],
                        'count': row[2],
                        'start_date': row[3],
                        'end_date': row[4]
                    })
                    total_rows += row[2]
            except Exception as e:
                print(f"Error getting summary for table {table}: {e}")
        
        # Convert to DataFrame
        if summary_rows:
            summary = pd.DataFrame(summary_rows)
        else:
            # Empty DataFrame with expected columns
            summary = pd.DataFrame(columns=['ticker', 'timeframe', 'count', 'start_date', 'end_date'])
        
        # Print the results
        print("\nIngested Data Summary:")
        print(f"{'Asset':<10} {'Timeframe':<10} {'Count':<10} {'Date Range':<30}")
        print("-" * 60)
        
        for _, row in summary.iterrows():
            ticker = row['ticker']
            timeframe = row['timeframe']
            count = row['count']
            date_range = f"{row['start_date']} to {row['end_date']}"
            print(f"{ticker:<10} {timeframe:<10} {count:<10} {date_range:<30}")
            
        print("-" * 60)
        print(f"{'Total':<21} {total_rows:<10}")
        
        # Log execution completion
        logger.log_execution_end(
            execution_id,
            "completed",
            f"Successfully ingested data for {len(CRYPTO_ASSETS) + len(STOCK_ASSETS)} assets across {len(TIMEFRAMES)} timeframes",
            {
                "total_rows": int(total_rows),
                "asset_count": len(CRYPTO_ASSETS) + len(STOCK_ASSETS),
                "timeframe_count": len(TIMEFRAMES),
                "asset_datasets": asset_datasets
            }
        )
        
    except Exception as e:
        print(f"Error ingesting data from taapi.io: {e}")
        
        # Log execution failure
        logger.log_execution_end(
            execution_id,
            "failed",
            f"Error: {str(e)}"
        )
    finally:
        # Close the database connection
        db_manager.close()

if __name__ == "__main__":
    main() 