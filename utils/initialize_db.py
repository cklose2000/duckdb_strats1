"""
Database Initialization Utility

This script initializes the DuckDB database and runs all migrations.
It creates the database file if it doesn't exist and applies all migration scripts.
"""

import os
import glob
import time
from pathlib import Path
from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger

def run_migration(db_manager, migration_path):
    """
    Run a single migration script.
    
    Args:
        db_manager (DBManager): Database manager instance.
        migration_path (str): Path to the migration script.
        
    Returns:
        float: Execution time in seconds.
    """
    print(f"Running migration: {migration_path}")
    
    # Read the migration script
    with open(migration_path, 'r') as f:
        migration_sql = f.read()
        
    # Execute the migration
    start_time = time.time()
    db_manager.conn.execute(migration_sql)
    execution_time = time.time() - start_time
    
    print(f"Migration {migration_path} completed in {execution_time:.2f} seconds")
    
    return execution_time

def init_database():
    """
    Initialize the database and run all migrations.
    
    Returns:
        DBManager: Database manager instance.
    """
    # Get project root and migrations directory
    root_dir = Path(__file__).parent.parent
    migrations_dir = root_dir / "migrations"
    db_path = root_dir / "db" / "backtest.ddb"
    
    # Create the database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create a database manager
    db_manager = DBManager(db_path)
    db_manager.connect()
    
    # Run all migrations in order
    migration_files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
    
    if not migration_files:
        print("No migration files found!")
        return db_manager
    
    print(f"Found {len(migration_files)} migration files")
    
    # Create a table to track migrations
    db_manager.conn.execute("""
    CREATE TABLE IF NOT EXISTS metadata_migrations (
        migration_id VARCHAR PRIMARY KEY,
        migration_name VARCHAR NOT NULL,
        migration_path VARCHAR NOT NULL,
        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        execution_time DOUBLE,
        status VARCHAR
    )
    """)
    
    # Run each migration file that hasn't been run yet
    for migration_path in migration_files:
        migration_name = os.path.basename(migration_path)
        
        # Check if the migration has already been run
        result = db_manager.conn.execute(
            "SELECT migration_id FROM metadata_migrations WHERE migration_path = ?",
            (migration_path,)
        ).fetchone()
        
        if result:
            print(f"Migration {migration_name} already executed, skipping")
            continue
            
        try:
            # Run the migration
            execution_time = run_migration(db_manager, migration_path)
            
            # Log the migration
            db_manager.conn.execute(
                """
                INSERT INTO metadata_migrations (
                    migration_id, migration_name, migration_path, 
                    executed_at, execution_time, status
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, 'success')
                """,
                (migration_name, migration_name, migration_path, execution_time)
            )
            
            # Commit the changes
            db_manager.conn.commit()
            
        except Exception as e:
            print(f"Error running migration {migration_name}: {e}")
            
            # Log the failed migration
            db_manager.conn.execute(
                """
                INSERT INTO metadata_migrations (
                    migration_id, migration_name, migration_path, 
                    executed_at, execution_time, status
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, NULL, 'failed')
                """,
                (migration_name, migration_name, migration_path)
            )
            
            # Commit the changes
            db_manager.conn.commit()
            
            # Re-raise the exception
            raise
    
    print("Database initialization completed successfully")
    
    return db_manager

if __name__ == "__main__":
    # Initialize the database
    db_manager = init_database()
    
    # Log the initialization in metadata
    try:
        logger = MetadataLogger(db_manager.conn)
        script_id = logger.log_script(
            __file__,
            description="Initialize database and run migrations",
            script_type="initialization"
        )
        logger.log_script_execution(script_id, status="success")
    except Exception as e:
        print(f"Error logging initialization script: {e}")
        
    # Close the database connection
    db_manager.close() 