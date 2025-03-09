"""
Backtest Report Generator

This script generates a comprehensive report of backtest results,
including strategy performance metrics, trade statistics, and visualizations.
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json
import traceback

# Set up the output directory
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_backtest_report():
    """Generate a comprehensive backtest report"""
    
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORTS_DIR, f"backtest_report_{report_time}.html")
    
    try:
        # Connect to the database
        con = duckdb.connect('db/backtest.ddb')
        print(f"Connected to database. Generating report...")
        
        # Start building the HTML report
        html = []
        html.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DuckDB Backtesting Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333366; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .metric-card { 
                    display: inline-block; 
                    width: 200px; 
                    margin: 10px; 
                    padding: 15px; 
                    border-radius: 5px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
                    text-align: center;
                    background-color: #f8f8f8;
                }
                .metric-value { font-size: 24px; font-weight: bold; color: #333366; }
                .metric-label { font-size: 14px; color: #666; }
                .chart-container { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>DuckDB Backtesting Report</h1>
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        """)
        
        # 1. Overall Summary
        html.append("<h2>Overall Summary</h2>")
        
        # Get summary statistics
        trade_stats = con.execute("""
        SELECT 
            COUNT(*) as total_trades, 
            COUNT(DISTINCT strategy_id) as total_strategies,
            ROUND(AVG(CASE WHEN trade_return IS NOT NULL THEN trade_return ELSE 0 END) * 100, 2) as avg_return_pct,
            ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) as win_rate
        FROM trade_log
        """).fetchone()
        
        # Add metric cards
        html.append("<div>")
        html.append(f"""
            <div class="metric-card">
                <div class="metric-value">{trade_stats[0]}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{trade_stats[1]}</div>
                <div class="metric-label">Strategies Tested</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{trade_stats[2]}%</div>
                <div class="metric-label">Average Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{trade_stats[3]}%</div>
                <div class="metric-label">Overall Win Rate</div>
            </div>
        """)
        html.append("</div>")
        
        # 2. Top Strategies
        html.append("<h2>Top Strategies by Sharpe Ratio</h2>")
        
        top_strategies = con.execute("""
        SELECT 
            strategy_id,
            total_trades,
            win_rate,
            profit_factor,
            ROUND(avg_return * 100, 2) as avg_return_pct,
            ROUND(max_drawdown * 100, 2) as max_drawdown_pct,
            ROUND(sharpe_ratio, 2) as sharpe_ratio,
            ROUND(annualized_return * 100, 2) as annualized_return_pct
        FROM strategy_performance_summary
        WHERE total_trades >= 2 AND sharpe_ratio IS NOT NULL
        ORDER BY sharpe_ratio DESC
        LIMIT 10
        """).fetchdf()
        
        if len(top_strategies) > 0:
            # Convert to HTML table
            html.append(top_strategies.to_html(index=False, classes="dataframe"))
            
            # Create a bar chart of Sharpe ratios
            plt.figure(figsize=(10, 6))
            sns.barplot(x='strategy_id', y='sharpe_ratio', data=top_strategies)
            plt.title('Top 10 Strategies by Sharpe Ratio')
            plt.xlabel('Strategy ID')
            plt.ylabel('Sharpe Ratio')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the chart
            chart_file = os.path.join(REPORTS_DIR, f"sharpe_ratio_chart_{report_time}.png")
            plt.savefig(chart_file)
            plt.close()
            
            # Add the chart to the report
            html.append(f"""
            <div class="chart-container">
                <img src="{os.path.basename(chart_file)}" alt="Sharpe Ratio Chart" style="max-width: 100%;">
            </div>
            """)
        else:
            html.append("<p>No strategies with valid Sharpe ratio found.</p>")
        
        # 3. Trade Analysis by Direction
        html.append("<h2>Trade Analysis by Direction</h2>")
        
        direction_stats = con.execute("""
        SELECT 
            direction, 
            COUNT(*) as count,
            ROUND(AVG(CASE WHEN trade_return IS NOT NULL THEN trade_return ELSE 0 END) * 100, 2) as avg_return_pct,
            ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) as win_rate
        FROM trade_log
        GROUP BY direction
        """).fetchdf()
        
        # Convert to HTML table
        html.append(direction_stats.to_html(index=False, classes="dataframe"))
        
        # Create a bar chart of returns by direction
        plt.figure(figsize=(8, 5))
        sns.barplot(x='direction', y='avg_return_pct', data=direction_stats)
        plt.title('Average Return by Direction')
        plt.xlabel('Direction')
        plt.ylabel('Average Return (%)')
        plt.tight_layout()
        
        # Save the chart
        chart_file = os.path.join(REPORTS_DIR, f"direction_return_chart_{report_time}.png")
        plt.savefig(chart_file)
        plt.close()
        
        # Add the chart to the report
        html.append(f"""
        <div class="chart-container">
            <img src="{os.path.basename(chart_file)}" alt="Direction Return Chart" style="max-width: 100%;">
        </div>
        """)
        
        # 4. Trade Distribution
        html.append("<h2>Trade Distribution</h2>")
        
        # Get trade returns for histogram
        trade_returns = con.execute("""
        SELECT trade_return * 100 as return_pct
        FROM trade_log
        WHERE trade_return IS NOT NULL
        """).fetchdf()
        
        if len(trade_returns) > 0:
            # Create a histogram of trade returns
            plt.figure(figsize=(10, 6))
            sns.histplot(trade_returns['return_pct'], bins=20, kde=True)
            plt.title('Distribution of Trade Returns')
            plt.xlabel('Return (%)')
            plt.ylabel('Frequency')
            plt.axvline(x=0, color='r', linestyle='--')
            plt.tight_layout()
            
            # Save the chart
            chart_file = os.path.join(REPORTS_DIR, f"return_distribution_chart_{report_time}.png")
            plt.savefig(chart_file)
            plt.close()
            
            # Add the chart to the report
            html.append(f"""
            <div class="chart-container">
                <img src="{os.path.basename(chart_file)}" alt="Return Distribution Chart" style="max-width: 100%;">
            </div>
            """)
        else:
            html.append("<p>No trade return data available for distribution analysis.</p>")
        
        # 5. System Metadata
        html.append("<h2>System Metadata</h2>")
        
        metadata_counts = con.execute("""
        SELECT 
            (SELECT COUNT(*) FROM metadata_scripts) as scripts,
            (SELECT COUNT(*) FROM metadata_queries) as queries,
            (SELECT COUNT(*) FROM metadata_models) as models,
            (SELECT COUNT(*) FROM metadata_executions) as executions,
            (SELECT COUNT(*) FROM metadata_datasets) as datasets
        """).fetchone()
        
        html.append(f"""
        <p>Metadata object counts:</p>
        <ul>
            <li>Scripts: {metadata_counts[0]}</li>
            <li>Queries: {metadata_counts[1]}</li>
            <li>Models: {metadata_counts[2]}</li>
            <li>Executions: {metadata_counts[3]}</li>
            <li>Datasets: {metadata_counts[4]}</li>
        </ul>
        """)
        
        # 6. Last Execution Details
        html.append("<h2>Last Execution Details</h2>")
        
        last_exec = con.execute("""
        SELECT 
            execution_id, 
            execution_name,
            description,
            start_time,
            end_time,
            status,
            result_summary
        FROM metadata_executions
        ORDER BY start_time DESC
        LIMIT 1
        """).fetchone()
        
        if last_exec:
            html.append(f"""
            <p>Last execution:</p>
            <ul>
                <li>ID: {last_exec[0]}</li>
                <li>Name: {last_exec[1]}</li>
                <li>Description: {last_exec[2]}</li>
                <li>Start: {last_exec[3]}</li>
                <li>End: {last_exec[4] or 'N/A'}</li>
                <li>Status: {last_exec[5]}</li>
                <li>Summary: {last_exec[6] or 'N/A'}</li>
            </ul>
            """)
        
        # Close the HTML
        html.append("""
        </body>
        </html>
        """)
        
        # Write the report to file
        with open(report_file, 'w') as f:
            f.write('\n'.join(html))
        
        print(f"Report generated successfully: {report_file}")
        
        # Close the connection
        con.close()
        
        return report_file
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    generate_backtest_report() 