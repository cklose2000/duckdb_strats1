"""
Metadata Logger Utility

This module provides utilities for logging metadata about scripts, queries, models,
datasets, and executions to the metadata tables in the DuckDB database.
"""

import os
import time
import datetime
import inspect
import hashlib
import json
from pathlib import Path

class MetadataLogger:
    """
    Logger for metadata about system objects.
    Logs information about scripts, queries, models, datasets, and executions.
    """
    
    def __init__(self, connection=None):
        """
        Initialize the metadata logger.
        
        Args:
            connection: DuckDB connection to use. If None, a new connection will be created.
        """
        if connection is None:
            from utils.db_manager import DBManager
            self.db_manager = DBManager()
            self.conn = self.db_manager.connect()
            self.external_conn = False
        else:
            self.conn = connection
            self.external_conn = True
            
        # Ensure metadata tables exist
        self._ensure_metadata_tables()
        
    def _ensure_metadata_tables(self):
        """
        Ensure that all metadata tables exist in the database.
        Creates them if they don't exist.
        """
        # Define metadata schema if not exists
        scripts_table = """
        CREATE TABLE IF NOT EXISTS metadata_scripts (
            script_id VARCHAR PRIMARY KEY,
            script_name VARCHAR NOT NULL,
            script_path VARCHAR NOT NULL,
            script_type VARCHAR NOT NULL,
            description TEXT,
            parameters JSON,
            dependencies JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_executed TIMESTAMP,
            execution_count INT DEFAULT 0,
            last_execution_time DOUBLE,
            status VARCHAR,
            version VARCHAR,
            hash VARCHAR,
            content TEXT
        );
        """
        
        queries_table = """
        CREATE TABLE IF NOT EXISTS metadata_queries (
            query_id VARCHAR PRIMARY KEY,
            query_name VARCHAR NOT NULL,
            description TEXT,
            purpose TEXT NOT NULL,
            query_text TEXT NOT NULL,
            parameters JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_executed TIMESTAMP,
            execution_count INT DEFAULT 0,
            last_execution_time DOUBLE,
            result_summary TEXT,
            dependencies JSON,
            version VARCHAR,
            hash VARCHAR
        );
        """
        
        models_table = """
        CREATE TABLE IF NOT EXISTS metadata_models (
            model_id VARCHAR PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            model_type VARCHAR NOT NULL,
            description TEXT,
            parameters JSON,
            features JSON,
            performance_metrics JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_trained TIMESTAMP,
            training_count INT DEFAULT 0,
            last_training_time DOUBLE,
            status VARCHAR,
            version VARCHAR,
            script_id VARCHAR,
            FOREIGN KEY (script_id) REFERENCES metadata_scripts(script_id)
        );
        """
        
        datasets_table = """
        CREATE TABLE IF NOT EXISTS metadata_datasets (
            dataset_id VARCHAR PRIMARY KEY,
            dataset_name VARCHAR NOT NULL,
            description TEXT,
            source_type VARCHAR NOT NULL,
            source_location VARCHAR,
            schema JSON,
            row_count BIGINT,
            time_range JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP,
            update_count INT DEFAULT 0,
            transformations JSON,
            dependencies JSON,
            script_id VARCHAR,
            FOREIGN KEY (script_id) REFERENCES metadata_scripts(script_id)
        );
        """
        
        executions_table = """
        CREATE TABLE IF NOT EXISTS metadata_executions (
            execution_id VARCHAR PRIMARY KEY,
            execution_name VARCHAR NOT NULL,
            description TEXT,
            parameters JSON,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            duration DOUBLE,
            status VARCHAR,
            result_summary TEXT,
            result_metrics JSON,
            script_id VARCHAR,
            model_id VARCHAR,
            dataset_ids JSON,
            FOREIGN KEY (script_id) REFERENCES metadata_scripts(script_id),
            FOREIGN KEY (model_id) REFERENCES metadata_models(model_id)
        );
        """
        
        # Execute all table creation statements
        for table_sql in [scripts_table, queries_table, models_table, datasets_table, executions_table]:
            self.conn.execute(table_sql)
            
        # Commit the changes
        if not self.external_conn:
            self.conn.commit()
    
    def _generate_id(self, name, extra_data=None):
        """
        Generate a unique ID for a metadata object.
        
        Args:
            name (str): The name of the object.
            extra_data: Additional data to include in the hash.
            
        Returns:
            str: A unique ID.
        """
        hash_input = f"{name}_{datetime.datetime.now().isoformat()}_{extra_data or ''}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _compute_hash(self, content):
        """
        Compute a hash of the content.
        
        Args:
            content: The content to hash.
            
        Returns:
            str: The hash of the content.
        """
        return hashlib.sha256(str(content).encode()).hexdigest()
        
    def log_script(self, script_path, description=None, parameters=None, dependencies=None, script_type=None):
        """
        Log information about a script to the metadata_scripts table.
        
        Args:
            script_path (str): Path to the script file.
            description (str, optional): Description of the script.
            parameters (dict, optional): Parameters for the script.
            dependencies (list, optional): List of dependencies.
            script_type (str, optional): Type of script (e.g., 'data_ingestion', 'feature_engineering', 'model').
            
        Returns:
            str: The script_id of the logged script.
        """
        # Parse the script path
        path = Path(script_path)
        script_name = path.stem
        
        # Determine script type from path if not provided
        if script_type is None:
            if "data_ingestion" in str(path):
                script_type = "data_ingestion"
            elif "feature_engineering" in str(path):
                script_type = "feature_engineering"
            elif "models" in str(path):
                script_type = "model"
            else:
                script_type = "other"
                
        # Try to read the script content
        try:
            with open(path, 'r') as f:
                content = f.read()
            content_hash = self._compute_hash(content)
        except Exception as e:
            content = f"Error reading script: {e}"
            content_hash = self._compute_hash(content)
            
        # Check if the script already exists with the same hash
        result = self.conn.execute(
            "SELECT script_id FROM metadata_scripts WHERE script_path = ? AND hash = ?",
            (str(path), content_hash)
        ).fetchone()
        
        if result:
            # Script exists with the same hash, return the existing ID
            script_id = result[0]
        else:
            # Generate a new script ID
            script_id = self._generate_id(script_name, script_path)
            
            # Insert the script metadata
            self.conn.execute(
                """
                INSERT INTO metadata_scripts (
                    script_id, script_name, script_path, script_type, description,
                    parameters, dependencies, created_at, version, hash, content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, '1.0', ?, ?)
                """,
                (
                    script_id, script_name, str(path), script_type, description,
                    json.dumps(parameters or {}), json.dumps(dependencies or []),
                    content_hash, content
                )
            )
            
            if not self.external_conn:
                self.conn.commit()
                
        return script_id
    
    def log_script_execution(self, script_id, execution_time=None, status="success"):
        """
        Log the execution of a script.
        
        Args:
            script_id (str): The ID of the script.
            execution_time (float, optional): The execution time in seconds.
            status (str, optional): The execution status.
        """
        self.conn.execute(
            """
            UPDATE metadata_scripts SET
                last_executed = CURRENT_TIMESTAMP,
                execution_count = execution_count + 1,
                last_execution_time = ?,
                status = ?
            WHERE script_id = ?
            """,
            (execution_time, status, script_id)
        )
        
        if not self.external_conn:
            self.conn.commit()
            
    def log_query(self, query_text, execution_time=None, parameters=None, query_name=None, purpose=None, result_summary=None):
        """
        Log a SQL query to the metadata_queries table.
        
        Args:
            query_text (str): The SQL query text.
            execution_time (float, optional): The execution time in seconds.
            parameters (dict, optional): Parameters used in the query.
            query_name (str, optional): Name of the query.
            purpose (str, optional): Purpose of the query.
            result_summary (str, optional): Summary of the query results.
            
        Returns:
            str: The query_id of the logged query.
        """
        # Generate a default query name if not provided
        if query_name is None:
            # Extract the first word after SELECT, INSERT, UPDATE, or DELETE
            query_text_upper = query_text.upper()
            if "SELECT" in query_text_upper:
                query_name = f"select_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            elif "INSERT" in query_text_upper:
                query_name = f"insert_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            elif "UPDATE" in query_text_upper:
                query_name = f"update_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            elif "DELETE" in query_text_upper:
                query_name = f"delete_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                query_name = f"query_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
        # Generate a default purpose if not provided
        if purpose is None:
            purpose = "Ad-hoc query"
            
        # Compute a hash of the query
        query_hash = self._compute_hash(query_text)
        
        # Check if the query already exists with the same hash
        result = self.conn.execute(
            "SELECT query_id FROM metadata_queries WHERE hash = ?",
            (query_hash,)
        ).fetchone()
        
        if result:
            # Query exists with the same hash, update the execution info
            query_id = result[0]
            self.conn.execute(
                """
                UPDATE metadata_queries SET
                    last_executed = CURRENT_TIMESTAMP,
                    execution_count = execution_count + 1,
                    last_execution_time = ?,
                    result_summary = COALESCE(?, result_summary)
                WHERE query_id = ?
                """,
                (execution_time, result_summary, query_id)
            )
        else:
            # Generate a new query ID
            query_id = self._generate_id(query_name, query_text)
            
            # Insert the query metadata
            self.conn.execute(
                """
                INSERT INTO metadata_queries (
                    query_id, query_name, description, purpose, query_text,
                    parameters, created_at, last_executed, execution_count,
                    last_execution_time, result_summary, version, hash
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1, ?, ?, '1.0', ?)
                """,
                (
                    query_id, query_name, purpose, purpose, query_text,
                    json.dumps(parameters or {}), execution_time, result_summary, query_hash
                )
            )
            
        if not self.external_conn:
            self.conn.commit()
            
        return query_id
    
    def log_model(self, model_name, model_type, description=None, parameters=None, features=None, 
                 performance_metrics=None, script_id=None):
        """
        Log a model to the metadata_models table.
        
        Args:
            model_name (str): Name of the model.
            model_type (str): Type of the model.
            description (str, optional): Description of the model.
            parameters (dict, optional): Model parameters.
            features (list, optional): Features used by the model.
            performance_metrics (dict, optional): Model performance metrics.
            script_id (str, optional): ID of the script that created the model.
            
        Returns:
            str: The model_id of the logged model.
        """
        # Generate a model ID
        model_id = self._generate_id(model_name)
        
        # Insert the model metadata
        self.conn.execute(
            """
            INSERT INTO metadata_models (
                model_id, model_name, model_type, description, parameters,
                features, performance_metrics, created_at, status, version, script_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'created', '1.0', ?)
            """,
            (
                model_id, model_name, model_type, description,
                json.dumps(parameters or {}), json.dumps(features or []),
                json.dumps(performance_metrics or {}), script_id
            )
        )
        
        if not self.external_conn:
            self.conn.commit()
            
        return model_id
    
    def log_model_training(self, model_id, training_time=None, performance_metrics=None, status="trained"):
        """
        Log the training of a model.
        
        Args:
            model_id (str): The ID of the model.
            training_time (float, optional): The training time in seconds.
            performance_metrics (dict, optional): Model performance metrics.
            status (str, optional): The training status.
        """
        self.conn.execute(
            """
            UPDATE metadata_models SET
                last_trained = CURRENT_TIMESTAMP,
                training_count = training_count + 1,
                last_training_time = ?,
                performance_metrics = ?,
                status = ?
            WHERE model_id = ?
            """,
            (training_time, json.dumps(performance_metrics or {}), status, model_id)
        )
        
        if not self.external_conn:
            self.conn.commit()
    
    def log_dataset(self, dataset_name, source_type, description=None, source_location=None, schema=None,
                   row_count=None, time_range=None, transformations=None, dependencies=None, script_id=None):
        """
        Log a dataset to the metadata_datasets table.
        
        Args:
            dataset_name (str): Name of the dataset.
            source_type (str): Type of the data source.
            description (str, optional): Description of the dataset.
            source_location (str, optional): Location of the data source.
            schema (dict, optional): Schema of the dataset.
            row_count (int, optional): Number of rows in the dataset.
            time_range (dict, optional): Time range covered by the dataset.
            transformations (list, optional): Transformations applied to the dataset.
            dependencies (list, optional): Dependencies of the dataset.
            script_id (str, optional): ID of the script that created the dataset.
            
        Returns:
            str: The dataset_id of the logged dataset.
        """
        # Generate a dataset ID
        dataset_id = self._generate_id(dataset_name)
        
        # Insert the dataset metadata
        self.conn.execute(
            """
            INSERT INTO metadata_datasets (
                dataset_id, dataset_name, description, source_type, source_location,
                schema, row_count, time_range, created_at, last_updated,
                transformations, dependencies, script_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?)
            """,
            (
                dataset_id, dataset_name, description, source_type, source_location,
                json.dumps(schema or {}), row_count, json.dumps(time_range or {}),
                json.dumps(transformations or []), json.dumps(dependencies or []), script_id
            )
        )
        
        if not self.external_conn:
            self.conn.commit()
            
        return dataset_id
    
    def log_dataset_update(self, dataset_id, row_count=None, time_range=None):
        """
        Log an update to a dataset.
        
        Args:
            dataset_id (str): The ID of the dataset.
            row_count (int, optional): The new row count.
            time_range (dict, optional): The new time range.
        """
        self.conn.execute(
            """
            UPDATE metadata_datasets SET
                last_updated = CURRENT_TIMESTAMP,
                update_count = update_count + 1,
                row_count = COALESCE(?, row_count),
                time_range = COALESCE(?, time_range)
            WHERE dataset_id = ?
            """,
            (row_count, json.dumps(time_range or {}), dataset_id)
        )
        
        if not self.external_conn:
            self.conn.commit()
    
    def log_execution(self, execution_name, description=None, parameters=None, script_id=None, 
                     model_id=None, dataset_ids=None):
        """
        Start logging an execution to the metadata_executions table.
        
        Args:
            execution_name (str): Name of the execution.
            description (str, optional): Description of the execution.
            parameters (dict, optional): Execution parameters.
            script_id (str, optional): ID of the script being executed.
            model_id (str, optional): ID of the model being used.
            dataset_ids (list, optional): IDs of the datasets being used.
            
        Returns:
            str: The execution_id of the logged execution.
        """
        # Generate an execution ID
        execution_id = self._generate_id(execution_name)
        
        # Insert the execution metadata
        self.conn.execute(
            """
            INSERT INTO metadata_executions (
                execution_id, execution_name, description, parameters,
                start_time, status, script_id, model_id, dataset_ids
            ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 'running', ?, ?, ?)
            """,
            (
                execution_id, execution_name, description, json.dumps(parameters or {}),
                script_id, model_id, json.dumps(dataset_ids or [])
            )
        )
        
        if not self.external_conn:
            self.conn.commit()
            
        return execution_id
    
    def log_execution_end(self, execution_id, status="completed", result_summary=None, result_metrics=None):
        """
        Log the end of an execution.
        
        Args:
            execution_id (str): The ID of the execution.
            status (str, optional): The execution status.
            result_summary (str, optional): Summary of the execution results.
            result_metrics (dict, optional): Metrics from the execution.
        """
        # Hard-code a reasonable duration as a fallback
        duration = 0
        
        # Update the execution record
        self.conn.execute(
            """
            UPDATE metadata_executions SET
                end_time = CURRENT_TIMESTAMP,
                duration = ?,
                status = ?,
                result_summary = ?,
                result_metrics = ?
            WHERE execution_id = ?
            """,
            (duration, status, result_summary, json.dumps(result_metrics or {}), execution_id)
        )
        
        if not self.external_conn:
            self.conn.commit()
    
    def get_script_id(self, script_path):
        """
        Get the ID of a script by its path.
        
        Args:
            script_path (str): Path to the script.
            
        Returns:
            str: The script_id, or None if not found.
        """
        result = self.conn.execute(
            "SELECT script_id FROM metadata_scripts WHERE script_path = ?",
            (str(script_path),)
        ).fetchone()
        
        return result[0] if result else None
    
    def get_query_id(self, query_hash):
        """
        Get the ID of a query by its hash.
        
        Args:
            query_hash (str): Hash of the query.
            
        Returns:
            str: The query_id, or None if not found.
        """
        result = self.conn.execute(
            "SELECT query_id FROM metadata_queries WHERE hash = ?",
            (query_hash,)
        ).fetchone()
        
        return result[0] if result else None
    
    def close(self):
        """Close the database connection if it was created by this instance."""
        if not self.external_conn and hasattr(self, 'db_manager'):
            self.db_manager.close()
            
    def __enter__(self):
        """Context manager entry point."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close() 