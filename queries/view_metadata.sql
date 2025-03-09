-- View Metadata Tables
-- This query provides a comprehensive view of the system's metadata

-- Parameters:
--   None

-- Part 1: Overview counts from each metadata table
WITH metadata_counts AS (
    SELECT 'Scripts' AS table_name, COUNT(*) AS count FROM metadata_scripts
    UNION ALL
    SELECT 'Queries' AS table_name, COUNT(*) AS count FROM metadata_queries
    UNION ALL
    SELECT 'Models' AS table_name, COUNT(*) AS count FROM metadata_models
    UNION ALL
    SELECT 'Datasets' AS table_name, COUNT(*) AS count FROM metadata_datasets
    UNION ALL
    SELECT 'Executions' AS table_name, COUNT(*) AS count FROM metadata_executions
),

-- Part 2: Recent script executions
recent_scripts AS (
    SELECT
        script_id,
        script_name,
        script_type,
        last_executed,
        execution_count,
        last_execution_time,
        status
    FROM metadata_scripts
    ORDER BY last_executed DESC
    LIMIT 10
),

-- Part 3: Recent executions
recent_executions AS (
    SELECT
        execution_id,
        execution_name,
        start_time,
        end_time,
        duration,
        status,
        result_summary
    FROM metadata_executions
    ORDER BY start_time DESC
    LIMIT 10
),

-- Part 4: Datasets with their transformations
dataset_info AS (
    SELECT
        dataset_id,
        dataset_name,
        source_type,
        row_count,
        created_at,
        last_updated,
        update_count,
        transformations
    FROM metadata_datasets
    ORDER BY last_updated DESC
),

-- Part 5: Model performance metrics
model_metrics AS (
    SELECT
        model_id,
        model_name,
        model_type,
        created_at,
        last_trained,
        training_count,
        status,
        performance_metrics
    FROM metadata_models
    ORDER BY last_trained DESC
)

-- Main query - Overview of metadata
SELECT 'Metadata Counts' AS section, table_name AS name, count AS value, NULL AS description, NULL AS timestamp
FROM metadata_counts

UNION ALL

-- Recent script executions
SELECT 
    'Recent Scripts' AS section, 
    script_name AS name, 
    CAST(execution_count AS VARCHAR) AS value, 
    script_type AS description, 
    last_executed AS timestamp
FROM recent_scripts

UNION ALL

-- Recent executions
SELECT 
    'Recent Executions' AS section,
    execution_name AS name,
    status AS value,
    result_summary AS description,
    end_time AS timestamp
FROM recent_executions

UNION ALL

-- Datasets
SELECT 
    'Datasets' AS section,
    dataset_name AS name,
    CAST(row_count AS VARCHAR) AS value,
    source_type AS description,
    last_updated AS timestamp
FROM dataset_info

UNION ALL

-- Models with metrics
SELECT 
    'Models' AS section,
    model_name AS name,
    status AS value,
    model_type AS description,
    last_trained AS timestamp
FROM model_metrics

ORDER BY 
    CASE section
        WHEN 'Metadata Counts' THEN 1
        WHEN 'Recent Scripts' THEN 2
        WHEN 'Recent Executions' THEN 3
        WHEN 'Datasets' THEN 4
        WHEN 'Models' THEN 5
        ELSE 6
    END,
    timestamp DESC; 