"""
Timeframe Data Ingestion Script

This script loads financial data CSV files with different timeframes into the DuckDB database.
It processes all CSV files in the data folder and creates appropriate tables for each timeframe.
All operations are logged in the metadata system.
"""

import os
import time
import pandas as pd
import glob
from pathlib import Path
import re

# Add parent directory to path to import from utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger

def ingest_timeframe_data(data_dir="data", db_manager=None, logger=None):
    """
    Load financial timeframe data from CSV files into the database.
    
    Args:
        data_dir (str): Path to the directory containing CSV files.
        db_manager (DBManager, optional): Database manager instance.
        logger (MetadataLogger, optional): Metadata logger instance.
        
    Returns:
        dict: Dictionary of dataset IDs mapped to timeframe names.
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
        description="Ingest financial data with various timeframes from CSV files",
        parameters={
            "data_dir": data_dir
        },
        script_type="data_ingestion"
    )
    
    start_time = time.time()
    
    # Get absolute path to data directory
    root_dir = Path(__file__).parent.parent.parent
    data_path = root_dir / data_dir
    
    # Find all CSV files
    csv_files = glob.glob(str(data_path / "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Dictionary to store dataset IDs
    dataset_ids = {}
    
    try:
        # Process each CSV file
        for csv_file in csv_files:
            file_name = os.path.basename(csv_file)
            print(f"Processing {file_name}...")
            
            # Extract ticker and timeframe from filename
            match = re.match(r'([A-Z]+)_([a-z0-9]+)_data\.csv', file_name)
            if not match:
                print(f"Skipping {file_name} - doesn't match expected naming pattern")
                continue
                
            ticker, timeframe = match.groups()
            
            # Table name based on timeframe
            table_name = f"price_data_{timeframe}"
            
            # Drop the table if it exists
            db_manager.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Create table with the structure we need
            create_table_query = f"""
            CREATE TABLE {table_name} (
                date TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                ticker VARCHAR,
                timeframe VARCHAR
            );
            """
            db_manager.execute(create_table_query)
            
            # Use DuckDB's direct CSV reading capability
            csv_import_query = f"""
            COPY {table_name} (date, open, high, low, close, volume)
            FROM '{csv_file}' (AUTO_DETECT TRUE);
            """
            db_manager.execute(csv_import_query)
            
            # Add ticker and timeframe
            update_query = f"""
            UPDATE {table_name}
            SET ticker = '{ticker}',
                timeframe = '{timeframe}';
            """
            db_manager.execute(update_query)
            
            # Get row count and time range
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            row_count = db_manager.conn.execute(count_query).fetchone()[0]
            
            time_range_query = f"""
            SELECT 
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM {table_name}
            """
            time_range_result = db_manager.conn.execute(time_range_query).fetchone()
            time_range = {
                "min_date": str(time_range_result[0]),
                "max_date": str(time_range_result[1])
            }
            
            # Log the dataset
            dataset_id = logger.log_dataset(
                dataset_name=table_name,
                source_type="CSV File",
                description=f"{ticker} {timeframe} price data",
                source_location=str(csv_file),
                schema={
                    "date": "TIMESTAMP",
                    "open": "DOUBLE",
                    "high": "DOUBLE",
                    "low": "DOUBLE",
                    "close": "DOUBLE",
                    "volume": "BIGINT",
                    "ticker": "VARCHAR",
                    "timeframe": "VARCHAR"
                },
                row_count=row_count,
                time_range=time_range,
                script_id=script_id
            )
            
            # Store the dataset ID
            dataset_ids[timeframe] = dataset_id
            
            print(f"  Loaded {row_count} rows for {ticker} {timeframe}")
            
        # Create a unified view that combines all timeframes
        # First check which tables actually exist
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
            try:
                db_manager.execute(create_unified_view_query)
                print("Created unified view across all timeframes")
            except Exception as e:
                print(f"Warning: Could not create unified view: {e}")
        
        # Log execution success
        execution_time = time.time() - start_time
        logger.log_script_execution(script_id, execution_time, "success")
        
        return dataset_ids
        
    except Exception as e:
        # Log execution failure
        execution_time = time.time() - start_time
        logger.log_script_execution(script_id, execution_time, f"failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Create DB manager and logger
    db_manager = DBManager()
    db_manager.connect()
    logger = MetadataLogger(db_manager.conn)
    
    try:
        # Log execution start
        execution_id = logger.log_execution(
            execution_name="timeframe_data_ingestion",
            description="Ingest price data across different timeframes",
            parameters={"data_dir": "data"}
        )
        
        # Ingest data
        dataset_ids = ingest_timeframe_data("data", db_manager, logger)
        
        # Build the query dynamically based on which tables were created
        table_exists_query = """
        SELECT table_name FROM information_schema.tables 
        WHERE table_name LIKE 'price_data_%'
        """
        existing_tables = db_manager.conn.execute(table_exists_query).fetchall()
        existing_tables = [t[0] for t in existing_tables]
        
        # Build the counts query dynamically
        count_parts = []
        for table in existing_tables:
            timeframe = table.replace("price_data_", "")
            count_parts.append(f"SELECT '{timeframe}' as timeframe, COUNT(*) as count FROM {table}")
        
        if count_parts:
            count_query = "WITH timeframe_counts AS (\n" + "\nUNION ALL\n".join(count_parts) + "\n) SELECT timeframe, count FROM timeframe_counts ORDER BY timeframe"
            
            result = db_manager.query_to_df(count_query)
            
            # Print the results
            print("\nIngested Data Summary:")
            print(f"{'Timeframe':<10} {'Count':<10}")
            print("-" * 20)
            
            total_rows = 0
            for _, row in result.iterrows():
                timeframe = row['timeframe']
                count = row['count']
                total_rows += count
                print(f"{timeframe:<10} {count:<10}")
                
            print("-" * 20)
            print(f"{'Total':<10} {total_rows:<10}")
            
            # Log execution completion
            logger.log_execution_end(
                execution_id,
                "completed",
                f"Successfully ingested data across {len(dataset_ids)} timeframes",
                {
                    "total_rows": int(total_rows),
                    "timeframe_count": len(dataset_ids)
                }
            )
        else:
            print("No tables were created")
            logger.log_execution_end(
                execution_id,
                "completed",
                "No tables were created",
                {
                    "total_rows": 0,
                    "timeframe_count": 0
                }
            )
        
    except Exception as e:
        print(f"Error ingesting timeframe data: {e}")
        
        # Log execution failure
        if 'execution_id' in locals():
            try:
                logger.log_execution_end(
                    execution_id,
                    "failed",
                    f"Error: {str(e)}"
                )
            except Exception as log_error:
                print(f"Additional error when logging execution end: {log_error}")
    finally:
        # Close the database connection
        db_manager.close() 