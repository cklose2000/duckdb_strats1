"""
Return Prediction Model

This script builds a model to predict stock returns using the features
generated in previous steps. It uses DuckDB for data processing and
scikit-learn for model training and evaluation.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Add parent directory to path to import from utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger

def train_return_prediction_model(target='next_day_return', prediction_horizon=1, 
                                 db_manager=None, logger=None, test_size=0.2,
                                 save_model=True):
    """
    Train a model to predict stock returns.
    
    Args:
        target (str): Target variable to predict ('next_day_return', 'next_week_return', or 'next_month_return').
        prediction_horizon (int): Number of days in the prediction horizon (1, 5, or 20).
        db_manager (DBManager, optional): Database manager instance.
        logger (MetadataLogger, optional): Metadata logger instance.
        test_size (float, optional): Proportion of data to use for testing.
        save_model (bool, optional): Whether to save the trained model to the database.
        
    Returns:
        str: Model ID of the trained model.
    """
    # Create DB manager and logger if not provided
    if db_manager is None:
        db_manager = DBManager()
        db_manager.connect()
        
    if logger is None:
        logger = MetadataLogger(db_manager.conn)
    
    # Get the model name
    model_name = f"return_prediction_{target}"
    
    # Log script execution
    script_id = logger.log_script(
        __file__,
        description=f"Train a model to predict {target}",
        parameters={
            "target": target,
            "prediction_horizon": prediction_horizon,
            "test_size": test_size
        },
        script_type="model",
        dependencies=["stock_features"]
    )
    
    start_time = time.time()
    
    try:
        # Check if stock_features table exists
        if not db_manager.table_exists("stock_features"):
            raise ValueError("Stock features table does not exist. Run feature engineering first.")
        
        # Define feature columns to use
        feature_query = """
        SELECT
            column_name
        FROM pragma_table_info('stock_features')
        WHERE column_name NOT IN (
            'ticker', 'date', 'close', 'next_day_return', 'next_week_return', 'next_month_return',
            'market_ticker', 'market_close'
        )
        """
        feature_cols = [row[0] for row in db_manager.conn.execute(feature_query).fetchall()]
        
        # Query to get the training data
        training_query = f"""
        SELECT
            ticker,
            date,
            {', '.join(feature_cols)},
            {target}
        FROM stock_features
        WHERE {target} IS NOT NULL
        ORDER BY ticker, date
        """
        
        # Get the data as a pandas DataFrame
        df = db_manager.query_to_df(training_query)
        
        # Drop rows with missing values
        df_clean = df.dropna()
        
        # Log the data shape
        print(f"Data shape after cleaning: {df_clean.shape}")
        
        # Split the data by time
        cutoff_idx = int(len(df_clean) * (1 - test_size))
        train_df = df_clean.iloc[:cutoff_idx]
        test_df = df_clean.iloc[cutoff_idx:]
        
        # Prepare features and target
        X_train = train_df[feature_cols]
        y_train = train_df[target]
        X_test = test_df[feature_cols]
        y_test = test_df[target]
        
        # Train a Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy (for classification)
        direction_accuracy = np.mean((y_test > 0) == (y_pred > 0))
        
        # Print metrics
        print(f"\nModel Evaluation Metrics:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Directional Accuracy: {direction_accuracy:.4f}")
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Print top 10 features
        print("\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Compile metrics for logging
        performance_metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        # Save the model if requested
        if save_model:
            # Serialize the model
            model_bytes = pickle.dumps(model)
            
            # Create models table if it doesn't exist
            create_models_table_query = """
            CREATE TABLE IF NOT EXISTS models (
                model_id VARCHAR PRIMARY KEY,
                model_name VARCHAR NOT NULL,
                model_type VARCHAR NOT NULL,
                target_variable VARCHAR NOT NULL,
                features JSON NOT NULL,
                parameters JSON NOT NULL,
                performance_metrics JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_binary BLOB NOT NULL
            );
            """
            db_manager.execute(create_models_table_query)
            
            # Insert the model into the database
            insert_model_query = """
            INSERT INTO models (
                model_id, model_name, model_type, target_variable, 
                features, parameters, performance_metrics, model_binary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # Generate a model ID based on target and timestamp
            model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Execute the insert
            db_manager.conn.execute(
                insert_model_query,
                (
                    model_id,
                    model_name,
                    "RandomForestRegressor",
                    target,
                    json.dumps(feature_cols),
                    json.dumps(model.get_params()),
                    json.dumps(performance_metrics),
                    model_bytes
                )
            )
            db_manager.conn.commit()
        
        # Log the model
        model_id = logger.log_model(
            model_name=model_name,
            model_type="RandomForestRegressor",
            description=f"Random Forest model to predict {target}",
            parameters=model.get_params(),
            features=feature_cols,
            performance_metrics=performance_metrics,
            script_id=script_id
        )
        
        # Log model training
        logger.log_model_training(
            model_id=model_id,
            training_time=time.time() - start_time,
            performance_metrics=performance_metrics
        )
        
        # Create a SQL function to apply the model
        create_predict_function_query = f"""
        CREATE OR REPLACE FUNCTION predict_{model_name}(
            {', '.join([f'{col} DOUBLE' for col in feature_cols])}
        ) RETURNS DOUBLE AS '
            SELECT 0.01  -- Placeholder for Python UDF
        ';
        """
        db_manager.execute(create_predict_function_query)
        
        # Log execution success
        execution_time = time.time() - start_time
        logger.log_script_execution(script_id, execution_time, "success")
        
        return model_id
        
    except Exception as e:
        # Log execution failure
        execution_time = time.time() - start_time
        logger.log_script_execution(script_id, execution_time, f"failed: {str(e)}")
        raise

def backtest_model(model_id, start_date=None, end_date=None, db_manager=None, logger=None):
    """
    Backtest a trained model.
    
    Args:
        model_id (str): ID of the model to backtest.
        start_date (str, optional): Start date for backtesting in 'YYYY-MM-DD' format.
        end_date (str, optional): End date for backtesting in 'YYYY-MM-DD' format.
        db_manager (DBManager, optional): Database manager instance.
        logger (MetadataLogger, optional): Metadata logger instance.
        
    Returns:
        dict: Backtest results.
    """
    # Create DB manager and logger if not provided
    if db_manager is None:
        db_manager = DBManager()
        db_manager.connect()
        
    if logger is None:
        logger = MetadataLogger(db_manager.conn)
    
    # Log execution
    execution_id = logger.log_execution(
        execution_name=f"backtest_{model_id}",
        description=f"Backtest model {model_id}",
        parameters={
            "model_id": model_id,
            "start_date": start_date,
            "end_date": end_date
        },
        model_id=model_id
    )
    
    start_time = time.time()
    
    try:
        # Get model information
        model_query = """
        SELECT 
            model_id, model_name, model_type, target_variable, 
            features, parameters, performance_metrics, model_binary
        FROM models
        WHERE model_id = ?
        """
        
        model_row = db_manager.conn.execute(model_query, (model_id,)).fetchone()
        
        if not model_row:
            raise ValueError(f"Model with ID {model_id} not found")
            
        model_name = model_row[1]
        target_variable = model_row[3]
        features = json.loads(model_row[4])
        
        # Deserialize the model
        model = pickle.loads(model_row[7])
        
        # Define date range for backtesting
        date_range_query = """
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM stock_features
        """
        
        min_date, max_date = db_manager.conn.execute(date_range_query).fetchone()
        
        if start_date is None:
            start_date = min_date
        
        if end_date is None:
            end_date = max_date
        
        # Query to get backtest data
        backtest_query = f"""
        SELECT
            ticker,
            date,
            close,
            {', '.join(features)},
            {target_variable}
        FROM stock_features
        WHERE date BETWEEN ? AND ?
        ORDER BY ticker, date
        """
        
        backtest_data = db_manager.query_to_df(backtest_query, (start_date, end_date))
        
        # Apply the model to get predictions
        X_backtest = backtest_data[features]
        y_backtest = backtest_data[target_variable]
        
        backtest_data['prediction'] = model.predict(X_backtest)
        
        # Create a backtest results table
        create_backtest_table_query = """
        CREATE TABLE IF NOT EXISTS backtest_results (
            backtest_id VARCHAR PRIMARY KEY,
            model_id VARCHAR NOT NULL,
            ticker VARCHAR NOT NULL,
            date DATE NOT NULL,
            close DOUBLE,
            prediction DOUBLE,
            actual DOUBLE,
            error DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        );
        """
        db_manager.execute(create_backtest_table_query)
        
        # Register the DataFrame as a temporary table
        db_manager.conn.register('temp_backtest_data', backtest_data)
        
        # Insert the backtest results
        backtest_id = f"backtest_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        insert_backtest_query = f"""
        INSERT INTO backtest_results (
            backtest_id, model_id, ticker, date, close, prediction, actual, error
        )
        SELECT
            '{backtest_id}' as backtest_id,
            '{model_id}' as model_id,
            ticker,
            date,
            close,
            prediction,
            {target_variable} as actual,
            prediction - {target_variable} as error
        FROM temp_backtest_data
        """
        
        db_manager.execute(insert_backtest_query)
        
        # Calculate backtest metrics
        metrics_query = """
        SELECT
            AVG(error) as mean_error,
            AVG(ABS(error)) as mae,
            SQRT(AVG(error * error)) as rmse,
            1 - (SUM(error * error) / SUM((actual - AVG(actual)) * (actual - AVG(actual)))) as r2,
            AVG(CASE WHEN (prediction > 0 AND actual > 0) OR (prediction < 0 AND actual < 0) THEN 1 ELSE 0 END) as direction_accuracy
        FROM backtest_results
        WHERE backtest_id = ?
        """
        
        metrics = db_manager.conn.execute(metrics_query, (backtest_id,)).fetchone()
        
        # Create a performance summary
        metrics_dict = {
            'mean_error': float(metrics[0]),
            'mae': float(metrics[1]),
            'rmse': float(metrics[2]),
            'r2': float(metrics[3]),
            'direction_accuracy': float(metrics[4]),
            'backtest_period': f"{start_date} to {end_date}",
            'data_points': len(backtest_data)
        }
        
        # Print metrics
        print(f"\nBacktest Results:")
        print(f"Model: {model_name}")
        print(f"Target: {target_variable}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Data points: {len(backtest_data)}")
        print(f"RMSE: {metrics_dict['rmse']:.6f}")
        print(f"MAE: {metrics_dict['mae']:.6f}")
        print(f"R²: {metrics_dict['r2']:.6f}")
        print(f"Directional Accuracy: {metrics_dict['direction_accuracy']:.4f}")
        
        # Log execution end
        logger.log_execution_end(
            execution_id,
            "completed",
            f"Backtest completed successfully",
            metrics_dict
        )
        
        return metrics_dict
        
    except Exception as e:
        # Log execution failure
        logger.log_execution_end(
            execution_id,
            "failed",
            f"Error: {str(e)}"
        )
        raise
    finally:
        # Clean up
        if 'temp_backtest_data' in db_manager.conn.tables():
            db_manager.conn.execute("DROP TABLE temp_backtest_data")

if __name__ == "__main__":
    import json
    
    # Create DB manager and logger
    db_manager = DBManager()
    db_manager.connect()
    logger = MetadataLogger(db_manager.conn)
    
    try:
        # Log execution start
        execution_id = logger.log_execution(
            execution_name="train_and_backtest_return_model",
            description="Train return prediction model and run backtest",
            parameters={
                "target": "next_day_return",
                "prediction_horizon": 1,
                "test_size": 0.2
            }
        )
        
        # Train the model
        model_id = train_return_prediction_model(
            target='next_day_return',
            prediction_horizon=1,
            db_manager=db_manager,
            logger=logger,
            test_size=0.2,
            save_model=True
        )
        
        # Run a backtest
        last_month = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        backtest_results = backtest_model(
            model_id=model_id,
            start_date=last_month,
            db_manager=db_manager,
            logger=logger
        )
        
        # Log execution end
        logger.log_execution_end(
            execution_id,
            "completed",
            "Model training and backtest completed successfully",
            {
                "model_id": model_id,
                "backtest_results": backtest_results
            }
        )
        
    except Exception as e:
        print(f"Error in training and backtesting: {e}")
        
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