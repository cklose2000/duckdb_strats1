"""
Database Manager Utility

This module provides utilities for connecting to and managing the DuckDB database.
It includes functions for establishing connections, executing queries, and managing transactions.
"""

import os
import time
import duckdb
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent
DB_PATH = ROOT_DIR / "db" / "backtest.ddb"

class DBManager:
    """
    Database connection manager for DuckDB.
    Handles connections, query execution, and transactions.
    """
    
    def __init__(self, db_path=None, read_only=False):
        """
        Initialize the database manager.
        
        Args:
            db_path (str, optional): Path to the database file. Defaults to DB_PATH.
            read_only (bool, optional): Whether to open the connection in read-only mode.
        """
        self.db_path = db_path or DB_PATH
        self.read_only = read_only
        self.conn = None
        
    def __enter__(self):
        """Context manager entry point."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()
        
    def connect(self):
        """
        Establish a connection to the DuckDB database.
        Creates the database file if it doesn't exist.
        """
        # Make sure the database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to the database
        self.conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
        return self.conn
    
    def close(self):
        """Close the database connection if it's open."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def execute(self, query, params=None, auto_commit=True):
        """
        Execute a SQL query.
        
        Args:
            query (str): The SQL query to execute.
            params (dict, optional): Parameters for the query.
            auto_commit (bool, optional): Whether to auto-commit the transaction.
            
        Returns:
            DuckDB result object.
        """
        if not self.conn:
            self.connect()
            
        start_time = time.time()
        result = self.conn.execute(query, params or {})
        execution_time = time.time() - start_time
        
        if auto_commit:
            self.conn.commit()
            
        # Log the query execution
        from utils.metadata_logger import MetadataLogger
        try:
            logger = MetadataLogger(self.conn)
            logger.log_query(query, execution_time, params)
        except Exception as e:
            # During initialization, the metadata tables might not exist yet
            print(f"Could not log query: {e}")
            
        return result
    
    def begin_transaction(self):
        """Begin a new transaction."""
        if not self.conn:
            self.connect()
        self.conn.begin()
        
    def commit(self):
        """Commit the current transaction."""
        if self.conn:
            self.conn.commit()
            
    def rollback(self):
        """Rollback the current transaction."""
        if self.conn:
            self.conn.rollback()
            
    def query_to_df(self, query, params=None):
        """
        Execute a query and return the results as a pandas DataFrame.
        
        Args:
            query (str): The SQL query to execute.
            params (dict, optional): Parameters for the query.
            
        Returns:
            pandas DataFrame with the query results.
        """
        if not self.conn:
            self.connect()
            
        return self.conn.execute(query, params or {}).fetchdf()
    
    def table_exists(self, table_name):
        """
        Check if a table exists in the database.
        
        Args:
            table_name (str): Name of the table to check.
            
        Returns:
            bool: True if the table exists, False otherwise.
        """
        if not self.conn:
            self.connect()
            
        result = self.conn.execute(f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'").fetchone()[0]
        return result > 0

# Convenience function to get a database connection
def get_connection(read_only=False):
    """
    Get a database connection.
    
    Args:
        read_only (bool, optional): Whether to open the connection in read-only mode.
        
    Returns:
        DuckDB connection object.
    """
    return DBManager(read_only=read_only).connect() 