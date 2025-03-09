"""
Auto Discovery Utility

This module provides utilities for automatically discovering and cataloging
database objects, including tables, views, functions, and more.
"""

import json
from utils.db_manager import DBManager

class AutoDiscovery:
    """
    Utility for discovering database objects.
    Provides methods to catalog tables, views, functions, models, and transformations.
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize the auto discovery utility.
        
        Args:
            db_manager (DBManager, optional): Database manager instance.
        """
        self.db_manager = db_manager or DBManager()
        if not hasattr(self.db_manager, 'conn') or self.db_manager.conn is None:
            self.db_manager.connect()
    
    def discover_tables_and_views(self):
        """
        Discover all tables and views in the database.
        
        Returns:
            dict: Dictionary containing all tables and views, with their schemas.
        """
        # Query information_schema.tables to find all tables and views
        query = """
        SELECT 
            table_name,
            table_type,
            array_agg(STRUCT_PACK(
                column_name,
                data_type,
                is_nullable,
                column_default
            ) ORDER BY ordinal_position) as columns
        FROM information_schema.tables
        JOIN information_schema.columns USING (table_name)
        WHERE table_schema = 'main'
        GROUP BY table_name, table_type
        ORDER BY table_name
        """
        
        result = self.db_manager.query_to_df(query)
        
        tables_and_views = {}
        
        for _, row in result.iterrows():
            table_name = row['table_name']
            table_type = row['table_type']
            columns = row['columns']
            
            # Format columns
            formatted_columns = []
            for col in columns:
                formatted_columns.append({
                    'name': col['column_name'],
                    'type': col['data_type'],
                    'nullable': col['is_nullable'] == 'YES',
                    'default': col['column_default']
                })
            
            # Store in the dictionary
            tables_and_views[table_name] = {
                'type': table_type,
                'columns': formatted_columns
            }
            
            # If it's a metadata table, get the count
            if table_name.startswith('metadata_'):
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_result = self.db_manager.conn.execute(count_query).fetchone()
                tables_and_views[table_name]['row_count'] = count_result[0]
        
        return tables_and_views
    
    def discover_functions(self):
        """
        Discover all user-defined functions in the database.
        
        Returns:
            dict: Dictionary containing all functions with their parameters.
        """
        # Query information_schema.functions to find all user-defined functions
        query = """
        SELECT 
            function_name,
            array_agg(STRUCT_PACK(
                parameter_name,
                data_type,
                ordinal_position,
                parameter_default
            ) ORDER BY ordinal_position) as parameters,
            return_type
        FROM information_schema.functions
        LEFT JOIN information_schema.function_parameters USING (function_name)
        WHERE function_schema = 'main'
        GROUP BY function_name, return_type
        ORDER BY function_name
        """
        
        try:
            result = self.db_manager.query_to_df(query)
            
            functions = {}
            
            for _, row in result.iterrows():
                function_name = row['function_name']
                parameters = row['parameters']
                return_type = row['return_type']
                
                # Format parameters
                formatted_parameters = []
                if parameters and parameters[0]['parameter_name'] is not None:
                    for param in parameters:
                        formatted_parameters.append({
                            'name': param['parameter_name'],
                            'type': param['data_type'],
                            'position': param['ordinal_position'],
                            'default': param['parameter_default']
                        })
                
                # Store in the dictionary
                functions[function_name] = {
                    'parameters': formatted_parameters,
                    'return_type': return_type
                }
                
            return functions
        except Exception as e:
            print(f"Error discovering functions: {e}")
            return {}
    
    def discover_models(self):
        """
        Discover all statistical models in the database.
        
        Returns:
            dict: Dictionary containing all models with their metrics.
        """
        # Check if the metadata_models table exists
        if not self.db_manager.table_exists('metadata_models'):
            return {}
            
        # Query the metadata_models table
        query = """
        SELECT 
            model_id,
            model_name,
            model_type,
            description,
            parameters,
            features,
            performance_metrics,
            created_at,
            last_trained,
            training_count,
            status,
            version
        FROM metadata_models
        """
        
        result = self.db_manager.query_to_df(query)
        
        models = {}
        
        for _, row in result.iterrows():
            model_name = row['model_name']
            
            # Parse JSON fields
            parameters = json.loads(row['parameters']) if row['parameters'] else {}
            features = json.loads(row['features']) if row['features'] else []
            performance_metrics = json.loads(row['performance_metrics']) if row['performance_metrics'] else {}
            
            # Store in the dictionary
            models[model_name] = {
                'id': row['model_id'],
                'type': row['model_type'],
                'description': row['description'],
                'parameters': parameters,
                'features': features,
                'metrics': performance_metrics,
                'created_at': str(row['created_at']),
                'last_trained': str(row['last_trained']) if row['last_trained'] else None,
                'training_count': row['training_count'],
                'status': row['status'],
                'version': row['version']
            }
        
        return models
    
    def discover_feature_transformations(self):
        """
        Discover all feature transformations in the database.
        
        Returns:
            dict: Dictionary containing all feature transformations.
        """
        # Check if the metadata_datasets table exists
        if not self.db_manager.table_exists('metadata_datasets'):
            return {}
            
        # Query the metadata_datasets table
        query = """
        SELECT 
            dataset_id,
            dataset_name,
            transformations
        FROM metadata_datasets
        WHERE transformations IS NOT NULL
        AND transformations != '[]'
        """
        
        result = self.db_manager.query_to_df(query)
        
        transformations = {}
        
        for _, row in result.iterrows():
            dataset_name = row['dataset_name']
            
            # Parse JSON field
            dataset_transformations = json.loads(row['transformations']) if row['transformations'] else []
            
            # Add to dictionary
            transformations[dataset_name] = {
                'id': row['dataset_id'],
                'transformations': dataset_transformations
            }
        
        return transformations
    
    def discover_all(self):
        """
        Discover all database objects.
        
        Returns:
            dict: Dictionary containing all discovered objects.
        """
        discovery = {
            'tables_and_views': self.discover_tables_and_views(),
            'functions': self.discover_functions(),
            'models': self.discover_models(),
            'feature_transformations': self.discover_feature_transformations()
        }
        
        return discovery
    
    def print_discovery_summary(self, discovery=None):
        """
        Print a summary of discovered objects.
        
        Args:
            discovery (dict, optional): Discovery dictionary from discover_all().
        """
        if discovery is None:
            discovery = self.discover_all()
            
        tables_count = sum(1 for _, v in discovery['tables_and_views'].items() if v['type'] == 'BASE TABLE')
        views_count = sum(1 for _, v in discovery['tables_and_views'].items() if v['type'] == 'VIEW')
        functions_count = len(discovery['functions'])
        models_count = len(discovery['models'])
        transformations_count = len(discovery['feature_transformations'])
        
        print(f"=== Database Objects Discovery Summary ===")
        print(f"Tables: {tables_count}")
        print(f"Views: {views_count}")
        print(f"Functions: {functions_count}")
        print(f"Models: {models_count}")
        print(f"Feature Transformations: {transformations_count}")
        print(f"=======================================")
        
        # Print table details
        if tables_count > 0:
            print("\nTables:")
            for name, details in discovery['tables_and_views'].items():
                if details['type'] == 'BASE TABLE':
                    col_count = len(details['columns'])
                    row_count = details.get('row_count', 'N/A')
                    print(f"  - {name}: {col_count} columns, {row_count} rows")
                    
        # Print model details
        if models_count > 0:
            print("\nModels:")
            for name, details in discovery['models'].items():
                print(f"  - {name} ({details['type']}): {details['status']}")
                if details['metrics']:
                    metrics_str = ", ".join(f"{k}={v}" for k, v in details['metrics'].items())
                    print(f"    Metrics: {metrics_str}")

if __name__ == "__main__":
    # Create the auto discovery utility
    discovery = AutoDiscovery()
    
    # Discover all database objects
    all_objects = discovery.discover_all()
    
    # Print the discovery summary
    discovery.print_discovery_summary(all_objects) 