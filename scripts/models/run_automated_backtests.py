"""
Automated Backtesting Pipeline

This script runs the entire automated backtesting and strategy optimization pipeline:
1. First generates and tests 100 strategies against the SPY 5-minute data
2. Selects the top 10 strategies based on Sharpe ratio and profit factor
3. Applies genetic algorithms to evolve and improve these strategies
"""

import os
import time
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger
from scripts.models.batch_strategy_tester import batch_test_strategies
from scripts.models.genetic_optimizer import run_genetic_optimization

def run_automated_backtesting(ticker='SPY', 
                             num_strategies=100, 
                             population_size=50,
                             generations=10,
                             plot_top=10):
    """
    Run the entire automated backtesting and optimization pipeline.
    
    Args:
        ticker (str): Ticker symbol to test
        num_strategies (int): Number of random strategies to generate and test
        population_size (int): Size of the genetic algorithm population
        generations (int): Number of generations for genetic evolution
        plot_top (int): Number of top strategies to plot
    """
    start_time = time.time()
    
    # Initialize database connections
    db_manager = DBManager()
    db_manager.connect()
    logger = MetadataLogger(db_manager.conn)
    
    # Log the start of the automated pipeline
    pipeline_id = logger.log_script(
        __file__,
        description="Automated backtesting and strategy optimization pipeline",
        parameters={
            "ticker": ticker,
            "num_strategies": num_strategies,
            "population_size": population_size,
            "generations": generations
        },
        script_type="pipeline"
    )
    
    try:
        # Step 1: Batch test random strategies
        print("\n" + "="*80)
        print(f"STEP 1: Batch testing {num_strategies} random strategies for {ticker}")
        print("="*80)
        
        batch_results = batch_test_strategies(
            num_strategies=num_strategies,
            ticker=ticker,
            plot_top=plot_top
        )
        
        # Step 2: Genetic optimization of top strategies
        print("\n" + "="*80)
        print(f"STEP 2: Genetic optimization of top strategies for {ticker}")
        print("="*80)
        
        final_population = run_genetic_optimization(
            ticker=ticker,
            population_size=population_size,
            generations=generations,
            mutation_rate=0.2,
            crossover_rate=0.7,
            elite_ratio=0.1,
            fitness_metric="sharpe_ratio"
        )
        
        # Step 3: Create a combined summary view in the database
        print("\n" + "="*80)
        print("STEP 3: Creating summary views in the database")
        print("="*80)
        
        # Create a view for all strategies
        all_strategies_view = """
        CREATE OR REPLACE VIEW all_strategies_summary AS
        SELECT 
            strategy_id,
            total_trades,
            win_rate,
            profit_factor,
            avg_return,
            max_drawdown,
            sharpe_ratio,
            annualized_return,
            parameters,
            CASE 
                WHEN strategy_id < 1000 THEN 'batch'
                ELSE 'genetic'
            END AS strategy_type
        FROM strategy_performance_summary
        WHERE total_trades > 0
        """
        
        # Create a view for top strategies by different metrics
        top_strategies_view = """
        CREATE OR REPLACE VIEW top_strategies AS
        WITH ranked AS (
            SELECT 
                strategy_id,
                total_trades,
                win_rate,
                profit_factor,
                sharpe_ratio,
                annualized_return,
                parameters,
                CASE 
                    WHEN strategy_id < 1000 THEN 'batch'
                    ELSE 'genetic'
                END AS strategy_type,
                RANK() OVER (ORDER BY sharpe_ratio DESC) AS sharpe_rank,
                RANK() OVER (ORDER BY profit_factor DESC) AS profit_rank,
                RANK() OVER (ORDER BY win_rate DESC) AS win_rate_rank,
                RANK() OVER (ORDER BY annualized_return DESC) AS return_rank
            FROM strategy_performance_summary
            WHERE total_trades >= 5
        )
        SELECT * FROM ranked
        WHERE sharpe_rank <= 10 OR profit_rank <= 10
        ORDER BY sharpe_rank
        """
        
        # Execute the view creation queries
        db_manager.conn.execute(all_strategies_view)
        db_manager.conn.execute(top_strategies_view)
        
        # Display the top strategies
        top_strategies = db_manager.query_to_df("""
        SELECT 
            strategy_id, 
            strategy_type,
            total_trades,
            win_rate,
            profit_factor,
            sharpe_ratio,
            annualized_return,
            sharpe_rank,
            profit_rank
        FROM top_strategies
        ORDER BY sharpe_rank
        LIMIT 10
        """)
        
        print("\nTop 10 strategies by Sharpe ratio:")
        print(top_strategies)
        
        # Create a query SQL file for future analysis
        queries_dir = Path(__file__).parent.parent.parent / "queries"
        if not queries_dir.exists():
            queries_dir.mkdir(parents=True)
            
        with open(queries_dir / "analyze_backtest_results.sql", "w") as f:
            f.write("""
-- Top strategies by Sharpe ratio
SELECT 
    strategy_id, 
    strategy_type,
    total_trades,
    win_rate,
    profit_factor,
    sharpe_ratio,
    annualized_return,
    sharpe_rank,
    profit_rank
FROM top_strategies
ORDER BY sharpe_rank
LIMIT 10;

-- Top strategies by profit factor
SELECT 
    strategy_id, 
    strategy_type,
    total_trades,
    win_rate,
    profit_factor,
    sharpe_ratio,
    annualized_return,
    sharpe_rank,
    profit_rank
FROM top_strategies
ORDER BY profit_rank
LIMIT 10;

-- Correlation between metrics
SELECT 
    CORR(win_rate, sharpe_ratio) AS win_rate_sharpe_corr,
    CORR(profit_factor, sharpe_ratio) AS profit_sharpe_corr,
    CORR(win_rate, profit_factor) AS win_rate_profit_corr,
    CORR(annualized_return, sharpe_ratio) AS return_sharpe_corr,
    CORR(annualized_return, max_drawdown) AS return_drawdown_corr
FROM all_strategies_summary
WHERE total_trades >= 5;

-- Trade analysis for best strategy
SELECT 
    s.strategy_id,
    s.sharpe_ratio,
    s.profit_factor,
    t.date,
    t.close,
    t.trend,
    t.rsi_14,
    t.signal,
    t.trade_action,
    t.entry_price,
    t.exit_price,
    t.trade_return,
    t.cumulative_return
FROM top_strategies s
JOIN parameterized_backtest_results t ON s.strategy_id = t.strategy_id
WHERE s.sharpe_rank = 1
AND t.trade_action IS NOT NULL
ORDER BY t.date;
            """)
            
        # Log metrics into metadata for the pipeline
        total_elapsed = time.time() - start_time
        
        logger.log_execution_end(
            pipeline_id,
            status="success",
            result_summary=f"Tested {num_strategies} strategies and ran genetic optimization for {generations} generations",
            result_metrics=json.dumps({
                "strategies_tested": num_strategies,
                "genetic_generations": generations,
                "top_strategy_id": int(top_strategies.iloc[0]["strategy_id"]) if not top_strategies.empty else None,
                "elapsed_time": total_elapsed
            })
        )
        
        print(f"\nAutomated backtesting pipeline completed in {total_elapsed:.2f} seconds")
        print(f"Results are stored in the database and can be analyzed with SQL queries")
        print(f"SQL queries have been saved to {queries_dir / 'analyze_backtest_results.sql'}")
        
    except Exception as e:
        # Log failure
        logger.log_execution_end(
            pipeline_id,
            status="error",
            result_summary=f"Error in automated backtesting pipeline: {str(e)}",
            result_metrics=json.dumps({"error": str(e)})
        )
        print(f"Error in automated backtesting pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close database connection
        db_manager.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run automated backtesting and optimization pipeline')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol to test')
    parser.add_argument('--num', type=int, default=100, help='Number of random strategies to test')
    parser.add_argument('--population', type=int, default=50, help='Genetic algorithm population size')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations for genetic evolution')
    parser.add_argument('--plot-top', type=int, default=10, help='Number of top strategies to plot')
    
    args = parser.parse_args()
    
    # Run the automated backtesting pipeline
    run_automated_backtesting(
        ticker=args.ticker,
        num_strategies=args.num,
        population_size=args.population,
        generations=args.generations,
        plot_top=args.plot_top
    ) 