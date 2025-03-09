#!/usr/bin/env python
"""
Trade Logger CLI

Command-line interface for trade logging operations in the DuckDB-centric backtesting framework.
This script provides utilities for listing, counting, analyzing, deleting, and truncating trades.
"""

import argparse
import sys
import json
from datetime import datetime
import pandas as pd
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.trade_logger import TradeLogger, get_trade_logger
from utils.db_manager import DBManager

def setup_parser():
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Command-line interface for trade logging operations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List trades command
    list_parser = subparsers.add_parser("list", help="List trades from the trade log")
    list_parser.add_argument("--strategy-id", type=int, help="Filter by strategy ID")
    list_parser.add_argument("--execution-id", type=str, help="Filter by execution ID")
    list_parser.add_argument("--symbol", type=str, help="Filter by symbol")
    list_parser.add_argument("--start-date", type=str, help="Filter by start date (YYYY-MM-DD)")
    list_parser.add_argument("--end-date", type=str, help="Filter by end date (YYYY-MM-DD)")
    list_parser.add_argument("--limit", type=int, default=50, help="Limit the number of trades returned")
    list_parser.add_argument("--format", choices=["table", "csv", "json"], default="table", 
                           help="Output format (default: table)")
    list_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # Count trades command
    count_parser = subparsers.add_parser("count", help="Count trades in the trade log")
    count_parser.add_argument("--strategy-id", type=int, help="Filter by strategy ID")
    count_parser.add_argument("--execution-id", type=str, help="Filter by execution ID")
    count_parser.add_argument("--symbol", type=str, help="Filter by symbol")
    count_parser.add_argument("--start-date", type=str, help="Filter by start date (YYYY-MM-DD)")
    count_parser.add_argument("--end-date", type=str, help="Filter by end date (YYYY-MM-DD)")
    
    # Analyze trades command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze trades in the trade log")
    analyze_parser.add_argument("--strategy-id", type=int, help="Filter by strategy ID")
    analyze_parser.add_argument("--execution-id", type=str, help="Filter by execution ID")
    analyze_parser.add_argument("--symbol", type=str, help="Filter by symbol")
    analyze_parser.add_argument("--start-date", type=str, help="Filter by start date (YYYY-MM-DD)")
    analyze_parser.add_argument("--end-date", type=str, help="Filter by end date (YYYY-MM-DD)")
    analyze_parser.add_argument("--by", choices=["strategy", "direction", "exit-reason", "date"], 
                             default="strategy", help="Group by field for analysis")
    analyze_parser.add_argument("--format", choices=["table", "csv", "json"], default="table", 
                             help="Output format (default: table)")
    analyze_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # Delete trades command
    delete_parser = subparsers.add_parser("delete", help="Delete trades from the trade log")
    delete_parser.add_argument("--strategy-id", type=int, help="Filter by strategy ID")
    delete_parser.add_argument("--execution-id", type=str, help="Filter by execution ID")
    delete_parser.add_argument("--confirm", action="store_true", help="Confirm deletion without prompting")
    
    # Truncate trades command
    truncate_parser = subparsers.add_parser("truncate", help="Truncate the trade log (delete all trades)")
    truncate_parser.add_argument("--confirm", action="store_true", help="Confirm truncation without prompting")
    
    # Export trades command
    export_parser = subparsers.add_parser("export", help="Export trades to a file")
    export_parser.add_argument("--strategy-id", type=int, help="Filter by strategy ID")
    export_parser.add_argument("--execution-id", type=str, help="Filter by execution ID")
    export_parser.add_argument("--symbol", type=str, help="Filter by symbol")
    export_parser.add_argument("--start-date", type=str, help="Filter by start date (YYYY-MM-DD)")
    export_parser.add_argument("--end-date", type=str, help="Filter by end date (YYYY-MM-DD)")
    export_parser.add_argument("--format", choices=["csv", "json", "parquet"], default="csv", 
                            help="Output format (default: csv)")
    export_parser.add_argument("--output", type=str, required=True, help="Output file")
    
    # Import trades command
    import_parser = subparsers.add_parser("import", help="Import trades from a file")
    import_parser.add_argument("--strategy-id", type=int, required=True, help="Strategy ID for imported trades")
    import_parser.add_argument("--execution-id", type=str, help="Execution ID for imported trades (default: new UUID)")
    import_parser.add_argument("--format", choices=["csv", "json", "parquet"], default="csv", 
                            help="Input format (default: csv)")
    import_parser.add_argument("--input", type=str, required=True, help="Input file")
    
    return parser

def list_trades(args, trade_logger):
    """List trades from the trade log."""
    # Get trades from the database
    trades_df = trade_logger.get_trades(
        strategy_id=args.strategy_id,
        execution_id=args.execution_id,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Sort by date
    if not trades_df.empty and 'date' in trades_df.columns:
        trades_df = trades_df.sort_values('date')
    
    # Limit the number of trades
    if args.limit > 0:
        trades_df = trades_df.head(args.limit)
    
    # Select a subset of columns for display
    display_columns = [
        'trade_id', 'strategy_id', 'date', 'symbol', 'action', 'direction',
        'entry_price', 'exit_price', 'trade_return', 'exit_reason'
    ]
    
    # Filter columns that exist
    display_columns = [col for col in display_columns if col in trades_df.columns]
    
    # Calculate return percentage for display
    if 'trade_return' in trades_df.columns:
        trades_df['return_pct'] = trades_df['trade_return'] * 100
        display_columns.append('return_pct')
    
    # Output the trades
    if trades_df.empty:
        print("No trades found matching the criteria")
        return
    
    if args.format == "table":
        if args.output:
            with open(args.output, 'w') as f:
                f.write(trades_df[display_columns].to_string(index=False))
            print(f"Trades written to {args.output}")
        else:
            print(f"Found {len(trades_df)} trades:")
            print(trades_df[display_columns].to_string(index=False))
    
    elif args.format == "csv":
        if args.output:
            trades_df[display_columns].to_csv(args.output, index=False)
            print(f"Trades written to {args.output}")
        else:
            print(trades_df[display_columns].to_csv(index=False))
    
    elif args.format == "json":
        # Convert datetime to string for JSON serialization
        for col in trades_df.select_dtypes(include=['datetime64']).columns:
            trades_df[col] = trades_df[col].astype(str)
        
        if args.output:
            trades_df[display_columns].to_json(args.output, orient="records", date_format="iso")
            print(f"Trades written to {args.output}")
        else:
            print(trades_df[display_columns].to_json(orient="records", date_format="iso"))

def count_trades(args, trade_logger):
    """Count trades in the trade log."""
    # Get trades from the database
    trades_df = trade_logger.get_trades(
        strategy_id=args.strategy_id,
        execution_id=args.execution_id,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Build the filter description
    filters = []
    if args.strategy_id:
        filters.append(f"strategy_id={args.strategy_id}")
    if args.execution_id:
        filters.append(f"execution_id={args.execution_id}")
    if args.symbol:
        filters.append(f"symbol={args.symbol}")
    if args.start_date:
        filters.append(f"start_date={args.start_date}")
    if args.end_date:
        filters.append(f"end_date={args.end_date}")
    
    filter_desc = " AND ".join(filters) if filters else "no filters"
    
    # Output the count
    print(f"Found {len(trades_df)} trades matching the criteria: {filter_desc}")

def analyze_trades(args, trade_logger):
    """Analyze trades in the trade log."""
    # Get trades from the database
    trades_df = trade_logger.get_trades(
        strategy_id=args.strategy_id,
        execution_id=args.execution_id,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if trades_df.empty:
        print("No trades found matching the criteria")
        return
    
    # Perform analysis based on the grouping field
    if args.by == "strategy":
        # Group by strategy_id
        analysis = trades_df.groupby('strategy_id').agg({
            'trade_id': 'count',
            'trade_return': ['mean', 'sum', 'std', 'min', 'max']
        })
        
        # Calculate win rate
        win_rates = trades_df.groupby('strategy_id')['trade_return'].apply(
            lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
        )
        
        # Add win rate to the analysis
        analysis[('win_rate', '')] = win_rates
        
        # Flatten the column MultiIndex
        analysis.columns = ['_'.join(col).strip('_') for col in analysis.columns.values]
        
        # Rename columns for clarity
        analysis = analysis.rename(columns={
            'trade_id_count': 'total_trades',
            'trade_return_mean': 'avg_return',
            'trade_return_sum': 'total_return',
            'trade_return_std': 'return_std',
            'trade_return_min': 'min_return',
            'trade_return_max': 'max_return'
        })
        
        # Convert returns to percentages
        for col in ['avg_return', 'total_return', 'return_std', 'min_return', 'max_return']:
            if col in analysis.columns:
                analysis[col] = analysis[col] * 100
        
        # Reset index to make strategy_id a column
        analysis = analysis.reset_index()
        
        # Sort by total trades and average return
        analysis = analysis.sort_values(['total_trades', 'avg_return'], ascending=[False, False])
        
    elif args.by == "direction":
        # Group by direction
        analysis = trades_df.groupby('direction').agg({
            'trade_id': 'count',
            'trade_return': ['mean', 'sum', 'std', 'min', 'max']
        })
        
        # Calculate win rate
        win_rates = trades_df.groupby('direction')['trade_return'].apply(
            lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
        )
        
        # Add win rate to the analysis
        analysis[('win_rate', '')] = win_rates
        
        # Flatten the column MultiIndex
        analysis.columns = ['_'.join(col).strip('_') for col in analysis.columns.values]
        
        # Rename columns for clarity
        analysis = analysis.rename(columns={
            'trade_id_count': 'total_trades',
            'trade_return_mean': 'avg_return',
            'trade_return_sum': 'total_return',
            'trade_return_std': 'return_std',
            'trade_return_min': 'min_return',
            'trade_return_max': 'max_return'
        })
        
        # Convert returns to percentages
        for col in ['avg_return', 'total_return', 'return_std', 'min_return', 'max_return']:
            if col in analysis.columns:
                analysis[col] = analysis[col] * 100
        
        # Reset index to make direction a column
        analysis = analysis.reset_index()
        
    elif args.by == "exit-reason":
        # Group by exit_reason
        analysis = trades_df.groupby('exit_reason').agg({
            'trade_id': 'count',
            'trade_return': ['mean', 'sum', 'std', 'min', 'max']
        })
        
        # Calculate win rate
        win_rates = trades_df.groupby('exit_reason')['trade_return'].apply(
            lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
        )
        
        # Add win rate to the analysis
        analysis[('win_rate', '')] = win_rates
        
        # Flatten the column MultiIndex
        analysis.columns = ['_'.join(col).strip('_') for col in analysis.columns.values]
        
        # Rename columns for clarity
        analysis = analysis.rename(columns={
            'trade_id_count': 'total_trades',
            'trade_return_mean': 'avg_return',
            'trade_return_sum': 'total_return',
            'trade_return_std': 'return_std',
            'trade_return_min': 'min_return',
            'trade_return_max': 'max_return'
        })
        
        # Convert returns to percentages
        for col in ['avg_return', 'total_return', 'return_std', 'min_return', 'max_return']:
            if col in analysis.columns:
                analysis[col] = analysis[col] * 100
        
        # Reset index to make exit_reason a column
        analysis = analysis.reset_index()
        
        # Sort by total trades
        analysis = analysis.sort_values('total_trades', ascending=False)
        
    elif args.by == "date":
        # Make sure date is a datetime type
        if 'date' in trades_df.columns:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df['date_only'] = trades_df['date'].dt.date
            
            # Group by date
            analysis = trades_df.groupby('date_only').agg({
                'trade_id': 'count',
                'trade_return': ['mean', 'sum', 'std', 'min', 'max']
            })
            
            # Calculate win rate
            win_rates = trades_df.groupby('date_only')['trade_return'].apply(
                lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
            )
            
            # Add win rate to the analysis
            analysis[('win_rate', '')] = win_rates
            
            # Flatten the column MultiIndex
            analysis.columns = ['_'.join(col).strip('_') for col in analysis.columns.values]
            
            # Rename columns for clarity
            analysis = analysis.rename(columns={
                'trade_id_count': 'total_trades',
                'trade_return_mean': 'avg_return',
                'trade_return_sum': 'total_return',
                'trade_return_std': 'return_std',
                'trade_return_min': 'min_return',
                'trade_return_max': 'max_return'
            })
            
            # Convert returns to percentages
            for col in ['avg_return', 'total_return', 'return_std', 'min_return', 'max_return']:
                if col in analysis.columns:
                    analysis[col] = analysis[col] * 100
            
            # Reset index to make date a column
            analysis = analysis.reset_index()
            
            # Sort by date
            analysis = analysis.sort_values('date_only')
        else:
            print("Error: Date column not found in trade log")
            return
    
    # Output the analysis
    if args.format == "table":
        if args.output:
            with open(args.output, 'w') as f:
                f.write(analysis.to_string(index=False))
            print(f"Analysis written to {args.output}")
        else:
            print(f"Analysis by {args.by}:")
            print(analysis.to_string(index=False))
    
    elif args.format == "csv":
        if args.output:
            analysis.to_csv(args.output, index=False)
            print(f"Analysis written to {args.output}")
        else:
            print(analysis.to_csv(index=False))
    
    elif args.format == "json":
        # Convert datetime to string for JSON serialization
        for col in analysis.select_dtypes(include=['datetime64']).columns:
            analysis[col] = analysis[col].astype(str)
        
        if args.output:
            analysis.to_json(args.output, orient="records", date_format="iso")
            print(f"Analysis written to {args.output}")
        else:
            print(analysis.to_json(orient="records", date_format="iso"))

def delete_trades(args, trade_logger):
    """Delete trades from the trade log."""
    # Get confirmation if not provided
    if not args.confirm:
        filters = []
        if args.strategy_id:
            filters.append(f"strategy_id={args.strategy_id}")
        if args.execution_id:
            filters.append(f"execution_id={args.execution_id}")
            
        filter_desc = " AND ".join(filters) if filters else "ALL TRADES (this will delete all trades!)"
        
        confirm = input(f"Are you sure you want to delete trades matching: {filter_desc}? [y/N] ")
        if confirm.lower() != 'y':
            print("Deletion canceled")
            return
    
    # Delete the trades
    count = trade_logger.delete_trades(
        strategy_id=args.strategy_id,
        execution_id=args.execution_id
    )
    
    print(f"Deleted {count} trades")

def truncate_trades(args, trade_logger):
    """Truncate the trade log (delete all trades)."""
    # Get confirmation if not provided
    if not args.confirm:
        confirm = input("Are you sure you want to truncate the trade log? This will delete ALL trades! [y/N] ")
        if confirm.lower() != 'y':
            print("Truncation canceled")
            return
    
    # Truncate the trade log
    trade_logger.truncate_trade_log()
    
    print("Trade log truncated")

def export_trades(args, trade_logger):
    """Export trades to a file."""
    # Get trades from the database
    trades_df = trade_logger.get_trades(
        strategy_id=args.strategy_id,
        execution_id=args.execution_id,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if trades_df.empty:
        print("No trades found matching the criteria")
        return
    
    # Export the trades
    if args.format == "csv":
        trades_df.to_csv(args.output, index=False)
    elif args.format == "json":
        # Convert datetime to string for JSON serialization
        for col in trades_df.select_dtypes(include=['datetime64']).columns:
            trades_df[col] = trades_df[col].astype(str)
            
        trades_df.to_json(args.output, orient="records", date_format="iso")
    elif args.format == "parquet":
        try:
            trades_df.to_parquet(args.output, index=False)
        except ImportError:
            print("Error: The 'pyarrow' package is required for Parquet export")
            print("Please install it with: pip install pyarrow")
            return
    
    print(f"Exported {len(trades_df)} trades to {args.output}")

def import_trades(args, trade_logger):
    """Import trades from a file."""
    # Import the trades
    try:
        if args.format == "csv":
            trades_df = pd.read_csv(args.input)
        elif args.format == "json":
            trades_df = pd.read_json(args.input, orient="records")
        elif args.format == "parquet":
            try:
                trades_df = pd.read_parquet(args.input)
            except ImportError:
                print("Error: The 'pyarrow' package is required for Parquet import")
                print("Please install it with: pip install pyarrow")
                return
    except Exception as e:
        print(f"Error importing trades: {str(e)}")
        return
    
    if trades_df.empty:
        print("No trades found in the import file")
        return
    
    # Log the trades
    trade_ids = trade_logger.log_trades_from_dataframe(
        strategy_id=args.strategy_id,
        trades_df=trades_df,
        symbol=trades_df['symbol'].iloc[0] if 'symbol' in trades_df.columns else 'SPY'
    )
    
    print(f"Imported {len(trade_ids)} trades")

def main():
    """Main function."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize the trade logger
    db_manager = DBManager()
    trade_logger = TradeLogger(db_manager)
    
    try:
        # Execute the command
        if args.command == "list":
            list_trades(args, trade_logger)
        elif args.command == "count":
            count_trades(args, trade_logger)
        elif args.command == "analyze":
            analyze_trades(args, trade_logger)
        elif args.command == "delete":
            delete_trades(args, trade_logger)
        elif args.command == "truncate":
            truncate_trades(args, trade_logger)
        elif args.command == "export":
            export_trades(args, trade_logger)
        elif args.command == "import":
            import_trades(args, trade_logger)
        else:
            parser.print_help()
    finally:
        # Close the database connection
        trade_logger.close()

if __name__ == "__main__":
    main() 