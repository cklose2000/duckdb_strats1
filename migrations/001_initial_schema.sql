-- Initial Schema Migration
-- Creates the metadata tables for tracking system objects

-- Transaction to ensure all tables are created or none
BEGIN TRANSACTION;

-- Metadata Scripts Table
-- Stores all Python/R scripts with execution metadata
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

-- Metadata Queries Table
-- Stores all SQL queries with purpose and results summary
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

-- Metadata Models Table
-- Tracks algorithmic models with parameters and performance
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

-- Metadata Datasets Table
-- Catalogs all data sources and transformations
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

-- Metadata Executions Table
-- Logs all backtests with parameters and results
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

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_scripts_name ON metadata_scripts(script_name);
CREATE INDEX IF NOT EXISTS idx_scripts_type ON metadata_scripts(script_type);
CREATE INDEX IF NOT EXISTS idx_queries_name ON metadata_queries(query_name);
CREATE INDEX IF NOT EXISTS idx_models_name ON metadata_models(model_name);
CREATE INDEX IF NOT EXISTS idx_models_type ON metadata_models(model_type);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON metadata_datasets(dataset_name);
CREATE INDEX IF NOT EXISTS idx_executions_name ON metadata_executions(execution_name);
CREATE INDEX IF NOT EXISTS idx_executions_status ON metadata_executions(status);

-- Commit the transaction
COMMIT; 