"""
Technical Indicators Feature Engineering Script

This script creates technical indicators from raw stock data.
All transformations are performed using SQL in DuckDB.
"""

import os
import time
from pathlib import Path

# Add parent directory to path to import from utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger

def create_technical_indicators(db_manager=None, logger=None):
    """
    Create technical indicators from raw stock data.
    
    Args:
        db_manager (DBManager, optional): Database manager instance.
        logger (MetadataLogger, optional): Metadata logger instance.
        
    Returns:
        str: Dataset ID of the created features.
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
        description="Create technical indicators from raw stock data",
        script_type="feature_engineering",
        dependencies=["raw_stock_data"]
    )
    
    start_time = time.time()
    
    try:
        # Check if raw_stock_data table exists
        if not db_manager.table_exists("raw_stock_data"):
            raise ValueError("Raw stock data table does not exist. Run data ingestion first.")
        
        # Start a transaction
        db_manager.begin_transaction()
        
        # Create a technical indicators view
        tech_indicators_query = """
        CREATE OR REPLACE VIEW technical_indicators AS
        WITH price_data AS (
            SELECT
                ticker,
                date,
                open,
                high,
                low,
                close,
                volume,
                -- Simple Moving Averages
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS sma_10,
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma_50,
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS sma_200,
                
                -- Exponential Moving Averages (approximated using window functions)
                -- EMA = Price(t) * k + EMA(y) * (1 – k) where k = 2/(N+1)
                LAST_VALUE(close * 0.1 + LAG(close * 0.1 + LAG(close * 0.1 + LAG(close * 0.1 + LAG(close * 0.1 + 
                LAG(close * 0.1 + LAG(close * 0.1 + LAG(close * 0.1 + LAG(close * 0.1 + LAG(close, 1, close) * 0.1, 1, close) * 0.9, 
                1, close) * 0.9, 1, close) * 0.9, 1, close) * 0.9, 1, close) * 0.9, 1, close) * 0.9, 1, close) * 0.9, 1, close) * 0.9, 1, close) * 0.9 
                OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS ema_10,
                
                -- Volume indicators
                AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS volume_sma_10,
                volume / AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS relative_volume
            FROM raw_stock_data
        )
        SELECT
            ticker,
            date,
            close,
            sma_10,
            sma_20,
            sma_50,
            sma_200,
            ema_10,
            
            -- Price relative to moving averages
            close / sma_10 - 1 AS price_rel_sma_10,
            close / sma_20 - 1 AS price_rel_sma_20,
            close / sma_50 - 1 AS price_rel_sma_50,
            close / sma_200 - 1 AS price_rel_sma_200,
            
            -- Moving average crossovers
            CASE WHEN sma_10 > sma_20 THEN 1 ELSE 0 END AS sma_10_20_crossover,
            CASE WHEN sma_20 > sma_50 THEN 1 ELSE 0 END AS sma_20_50_crossover,
            CASE WHEN sma_50 > sma_200 THEN 1 ELSE 0 END AS sma_50_200_crossover,
            
            -- Volatility indicators
            (high - low) / close AS daily_range,
            STDDEV(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) / 
            AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volatility_20,
            
            -- Volume indicators
            volume,
            volume_sma_10,
            relative_volume,
            
            -- Momentum indicators
            close / LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return,
            close / LAG(close, 5) OVER (PARTITION BY ticker ORDER BY date) - 1 AS weekly_return,
            close / LAG(close, 20) OVER (PARTITION BY ticker ORDER BY date) - 1 AS monthly_return,
            
            -- RSI (Relative Strength Index) - 14 period
            100 - (100 / (1 + (
                SUM(CASE WHEN close > LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) THEN close - LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) ELSE 0 END) 
                OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) /
                SUM(CASE WHEN close < LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) THEN LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) - close ELSE 0 END) 
                OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
            ))) AS rsi_14,
            
            -- MACD (Moving Average Convergence Divergence)
            ema_10 - 
            LAST_VALUE(close * 0.05 + LAG(close * 0.05 + LAG(close * 0.05 + LAG(close * 0.05 + LAG(close * 0.05 + 
            LAG(close * 0.05 + LAG(close * 0.05 + LAG(close * 0.05 + LAG(close * 0.05 + LAG(close, 1, close) * 0.05, 
            1, close) * 0.95, 1, close) * 0.95, 1, close) * 0.95, 1, close) * 0.95, 1, close) * 0.95, 1, close) * 0.95, 
            1, close) * 0.95, 1, close) * 0.95, 1, close) * 0.95 
            OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS macd
        FROM price_data
        ORDER BY ticker, date;
        """
        db_manager.execute(tech_indicators_query)
        
        # Create a more comprehensive feature table
        feature_table_query = """
        CREATE TABLE IF NOT EXISTS stock_features AS
        SELECT
            t.ticker,
            t.date,
            t.close,
            -- Technical indicators
            t.sma_10,
            t.sma_20,
            t.sma_50,
            t.sma_200,
            t.price_rel_sma_10,
            t.price_rel_sma_20,
            t.price_rel_sma_50,
            t.price_rel_sma_200,
            t.sma_10_20_crossover,
            t.sma_20_50_crossover,
            t.sma_50_200_crossover,
            t.daily_range,
            t.volatility_20,
            t.volume,
            t.relative_volume,
            t.rsi_14,
            t.macd,
            
            -- Returns
            t.daily_return,
            t.weekly_return,
            t.monthly_return,
            
            -- Future returns (targets)
            LEAD(t.close, 1) OVER (PARTITION BY t.ticker ORDER BY t.date) / t.close - 1 AS next_day_return,
            LEAD(t.close, 5) OVER (PARTITION BY t.ticker ORDER BY t.date) / t.close - 1 AS next_week_return,
            LEAD(t.close, 20) OVER (PARTITION BY t.ticker ORDER BY t.date) / t.close - 1 AS next_month_return,
            
            -- Market relative performance
            r.ticker AS market_ticker,
            r.close AS market_close,
            r.daily_return AS market_daily_return,
            r.weekly_return AS market_weekly_return,
            r.monthly_return AS market_monthly_return,
            
            -- Alpha (excess return relative to market)
            t.daily_return - r.daily_return AS daily_alpha,
            t.weekly_return - r.weekly_return AS weekly_alpha,
            t.monthly_return - r.monthly_return AS monthly_alpha
        FROM technical_indicators t
        LEFT JOIN technical_indicators r ON t.date = r.date AND r.ticker = 'SPY'
        ORDER BY t.ticker, t.date;
        """
        db_manager.execute(feature_table_query)
        
        # Commit the transaction
        db_manager.commit()
        
        # Get row count and time range
        count_query = "SELECT COUNT(*) FROM stock_features"
        row_count = db_manager.conn.execute(count_query).fetchone()[0]
        
        time_range_query = """
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM stock_features
        """
        time_range_result = db_manager.conn.execute(time_range_query).fetchone()
        time_range = {
            "min_date": str(time_range_result[0]),
            "max_date": str(time_range_result[1])
        }
        
        # Get feature stats
        feature_stats_query = """
        SELECT
            COUNT(DISTINCT ticker) AS ticker_count,
            AVG(volatility_20) AS avg_volatility,
            AVG(rsi_14) AS avg_rsi
        FROM stock_features
        """
        feature_stats = db_manager.conn.execute(feature_stats_query).fetchone()
        
        # Log the dataset
        transformations = [
            {
                "name": "technical_indicators",
                "type": "view",
                "description": "Common technical indicators for stock analysis"
            },
            {
                "name": "moving_averages",
                "type": "feature",
                "description": "Simple and exponential moving averages"
            },
            {
                "name": "relative_strength",
                "type": "feature",
                "description": "RSI indicator"
            },
            {
                "name": "volatility",
                "type": "feature",
                "description": "Volatility measures"
            }
        ]
        
        dataset_id = logger.log_dataset(
            dataset_name="stock_features",
            source_type="Derived",
            description="Technical indicators and features derived from raw stock data",
            source_location="raw_stock_data",
            schema={
                "ticker": "VARCHAR",
                "date": "DATE",
                "close": "DOUBLE",
                "sma_10": "DOUBLE",
                "sma_20": "DOUBLE",
                "sma_50": "DOUBLE",
                "sma_200": "DOUBLE",
                "price_rel_sma_10": "DOUBLE",
                "rsi_14": "DOUBLE",
                "macd": "DOUBLE",
                "daily_return": "DOUBLE",
                "next_day_return": "DOUBLE"
            },
            row_count=row_count,
            time_range=time_range,
            transformations=transformations,
            dependencies=["raw_stock_data"],
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
        
        # Rollback the transaction if it's still active
        db_manager.rollback()
        
        raise

if __name__ == "__main__":
    # Create DB manager and logger
    db_manager = DBManager()
    db_manager.connect()
    logger = MetadataLogger(db_manager.conn)
    
    try:
        # Log execution start
        execution_id = logger.log_execution(
            execution_name="create_technical_indicators",
            description="Generate technical indicators from stock data",
            parameters={}
        )
        
        # Create technical indicators
        dataset_id = create_technical_indicators(db_manager, logger)
        
        # Get a summary of the features
        summary_query = """
        SELECT 
            'Overall' AS metric,
            COUNT(*) AS value,
            'Total rows' AS description
        FROM stock_features
        
        UNION ALL
        
        SELECT 
            'Tickers',
            COUNT(DISTINCT ticker),
            'Unique tickers'
        FROM stock_features
        
        UNION ALL
        
        SELECT 
            'Features',
            (SELECT COUNT(*) FROM pragma_table_info('stock_features')) - 3, -- Subtract ticker, date, close
            'Number of features'
        
        UNION ALL
        
        SELECT 
            'Date Range',
            DATEDIFF('day', MIN(date), MAX(date)),
            'Days covered'
        FROM stock_features;
        """
        
        summary = db_manager.query_to_df(summary_query)
        
        # Print the results
        print("\nFeature Engineering Summary:")
        print(f"{'Metric':<15} {'Value':<10} {'Description':<30}")
        print("-" * 55)
        
        for _, row in summary.iterrows():
            metric = row['metric']
            value = row['value']
            description = row['description']
            print(f"{metric:<15} {value:<10} {description:<30}")
            
        # Log execution completion
        logger.log_execution_end(
            execution_id,
            "completed",
            "Successfully created technical indicators and features",
            {
                "row_count": int(summary.loc[0, 'value']),
                "ticker_count": int(summary.loc[1, 'value']),
                "feature_count": int(summary.loc[2, 'value'])
            }
        )
        
    except Exception as e:
        print(f"Error creating technical indicators: {e}")
        
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