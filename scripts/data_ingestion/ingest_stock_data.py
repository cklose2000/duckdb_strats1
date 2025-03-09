"""
Stock Data Ingestion Script

This script downloads historical stock data and loads it into the DuckDB database.
It logs all activities to the metadata tables and performs all transformations in SQL.
"""

import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import from utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger

def ingest_stock_data(tickers, start_date, end_date, db_manager=None, logger=None):
    """
    Download stock data and load it into the database.
    
    Args:
        tickers (list): List of ticker symbols to download.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        db_manager (DBManager, optional): Database manager instance.
        logger (MetadataLogger, optional): Metadata logger instance.
        
    Returns:
        str: Dataset ID of the ingested data.
    """
    # Create DB manager and logger if not provided
    if db_manager is None:
        db_manager = DBManager()
        db_manager.connect()
        
    if logger is None:
        logger = MetadataLogger(db_manager.conn)
    
    # Log script execution
    script_id = logger.log_script(
        __file__,
        description="Ingest historical stock data from Yahoo Finance",
        parameters={
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date
        },
        script_type="data_ingestion"
    )
    
    start_time = time.time()
    
    try:
        # Download data from Yahoo Finance
        stock_data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )
        
        # Process multi-level column DataFrame
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Multiple tickers
            data_frames = []
            for ticker in tickers:
                if ticker in stock_data.columns.levels[0]:
                    ticker_data = stock_data[ticker].copy()
                    ticker_data['ticker'] = ticker
                    ticker_data.reset_index(inplace=True)
                    data_frames.append(ticker_data)
            
            if data_frames:
                combined_data = pd.concat(data_frames, ignore_index=True)
            else:
                raise ValueError("No data available for the specified tickers")
        else:
            # Single ticker
            combined_data = stock_data.copy()
            combined_data['ticker'] = tickers[0]
            combined_data.reset_index(inplace=True)
        
        # Register the DataFrame as a temporary table
        db_manager.conn.register('temp_stock_data', combined_data)
        
        # Create the raw stock data table if it doesn't exist
        create_raw_table_query = """
        CREATE TABLE IF NOT EXISTS raw_stock_data (
            id INTEGER PRIMARY KEY,
            ticker VARCHAR NOT NULL,
            date DATE NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        );
        """
        db_manager.execute(create_raw_table_query)
        
        # Insert data into the raw_stock_data table
        insert_query = """
        INSERT INTO raw_stock_data (ticker, date, open, high, low, close, volume)
        SELECT 
            ticker,
            Date as date,
            Open as open,
            High as high,
            Low as low,
            Close as close,
            Volume as volume
        FROM temp_stock_data
        ON CONFLICT (ticker, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume;
        """
        db_manager.execute(insert_query)
        
        # Create a view with calculated returns
        create_returns_view_query = """
        CREATE OR REPLACE VIEW stock_returns AS
        SELECT
            ticker,
            date,
            close,
            LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) as prev_close,
            (close / LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) - 1) * 100 as daily_return,
            (close / LAG(close, 5) OVER (PARTITION BY ticker ORDER BY date) - 1) * 100 as weekly_return,
            (close / LAG(close, 20) OVER (PARTITION BY ticker ORDER BY date) - 1) * 100 as monthly_return
        FROM raw_stock_data;
        """
        db_manager.execute(create_returns_view_query)
        
        # Get row count and time range
        count_query = "SELECT COUNT(*) FROM raw_stock_data"
        row_count = db_manager.conn.execute(count_query).fetchone()[0]
        
        time_range_query = """
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM raw_stock_data
        """
        time_range_result = db_manager.conn.execute(time_range_query).fetchone()
        time_range = {
            "min_date": str(time_range_result[0]),
            "max_date": str(time_range_result[1])
        }
        
        # Log the dataset
        dataset_id = logger.log_dataset(
            dataset_name="raw_stock_data",
            source_type="Yahoo Finance",
            description=f"Historical stock data for {', '.join(tickers)}",
            source_location="yfinance API",
            schema={
                "ticker": "VARCHAR",
                "date": "DATE",
                "open": "DOUBLE",
                "high": "DOUBLE",
                "low": "DOUBLE",
                "close": "DOUBLE",
                "volume": "BIGINT"
            },
            row_count=row_count,
            time_range=time_range,
            transformations=[
                {
                    "name": "stock_returns",
                    "type": "view",
                    "description": "Calculated daily, weekly, and monthly returns"
                }
            ],
            script_id=script_id
        )
        
        # Log execution success
        execution_time = time.time() - start_time
        logger.log_script_execution(script_id, execution_time, "success")
        
        return dataset_id
        
    except Exception as e:
        # Log execution failure
        execution_time = time.time() - start_time
        logger.log_script_execution(script_id, execution_time, f"failed: {str(e)}")
        raise
    finally:
        # Clean up
        if 'temp_stock_data' in db_manager.conn.tables():
            db_manager.conn.execute("DROP TABLE temp_stock_data")

if __name__ == "__main__":
    # Default parameters
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Create DB manager and logger
    db_manager = DBManager()
    db_manager.connect()
    logger = MetadataLogger(db_manager.conn)
    
    try:
        # Log execution start
        execution_id = logger.log_execution(
            execution_name="stock_data_ingestion",
            description="Ingest historical stock data",
            parameters={
                "tickers": default_tickers,
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
        # Ingest stock data
        dataset_id = ingest_stock_data(
            default_tickers,
            start_date,
            end_date,
            db_manager,
            logger
        )
        
        # Query to count the number of rows per ticker
        count_query = """
        SELECT 
            ticker,
            COUNT(*) as row_count,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM raw_stock_data
        GROUP BY ticker
        ORDER BY ticker
        """
        
        result = db_manager.query_to_df(count_query)
        
        # Print the results
        print("\nIngested Stock Data Summary:")
        print(f"{'Ticker':<10} {'Count':<10} {'Date Range':<30}")
        print("-" * 50)
        
        for _, row in result.iterrows():
            ticker = row['ticker']
            count = row['row_count']
            date_range = f"{row['earliest_date']} to {row['latest_date']}"
            print(f"{ticker:<10} {count:<10} {date_range:<30}")
            
        # Log execution completion
        logger.log_execution_end(
            execution_id,
            "completed",
            f"Successfully ingested data for {len(default_tickers)} tickers",
            {
                "row_count": int(result['row_count'].sum()),
                "ticker_count": len(result),
                "date_range": f"{start_date} to {end_date}"
            }
        )
        
    except Exception as e:
        print(f"Error ingesting stock data: {e}")
        
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