"""
Genetic Strategy Optimizer

This module implements a genetic algorithm for evolving and optimizing trading strategies.
It takes the top-performing strategies from batch testing and evolves them through
mutation and crossover to find even better parameter combinations.
"""

import os
import time
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
import sys
import traceback

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger
from scripts.models.strategy_generator import StrategyParameters, generate_strategies
from scripts.models.parameterized_strategy import backtest_parameterized_strategy
from scripts.models.batch_strategy_tester import get_top_strategies, save_strategy_metrics

class GeneticOptimizer:
    """
    Class for optimizing trading strategies using genetic algorithms.
    """
    
    def __init__(self, 
                 population_size=50, 
                 generations=10, 
                 mutation_rate=0.2,
                 crossover_rate=0.7,
                 elite_ratio=0.1,
                 fitness_metric="sharpe_ratio",
                 ticker="SPY",
                 db_manager=None,
                 logger=None):
        """
        Initialize the genetic optimizer.
        
        Args:
            population_size (int): Size of the population in each generation
            generations (int): Number of generations to run
            mutation_rate (float): Probability of mutation (0-1)
            crossover_rate (float): Probability of crossover (0-1)
            elite_ratio (float): Ratio of top performers to keep (0-1)
            fitness_metric (str): Metric to use for fitness (sharpe_ratio, profit_factor)
            ticker (str): Ticker symbol to optimize for
            db_manager (DBManager, optional): Database manager instance
            logger (MetadataLogger, optional): Metadata logger instance
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.fitness_metric = fitness_metric
        self.ticker = ticker
        
        # Calculate the number of elites to keep
        self.num_elites = max(1, int(population_size * elite_ratio))
        
        # Strategy ID counter - start after batch strategies
        self.next_id = 1001
        
        # Create DB manager and logger if not provided
        if db_manager is None:
            self.db_manager = DBManager()
            self.db_manager.connect()
        else:
            self.db_manager = db_manager
            
        if logger is None:
            self.logger = MetadataLogger(self.db_manager.conn)
        else:
            self.logger = logger
            
        # Get date range
        date_query = """
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM price_data_5min
        WHERE ticker = 'SPY'
        """
        min_date, max_date = self.db_manager.conn.execute(date_query).fetchone()
        
        # Format dates as strings
        self.start_date = min_date.strftime('%Y-%m-%d')
        self.end_date = max_date.strftime('%Y-%m-%d')
        
        # Log initialization
        self.script_id = self.logger.log_script(
            __file__,
            description=f"Genetic optimization of trading strategies for {ticker}",
            parameters={
                "population_size": population_size,
                "generations": generations,
                "mutation_rate": mutation_rate,
                "crossover_rate": crossover_rate,
                "fitness_metric": fitness_metric,
                "ticker": ticker
            },
            script_type="genetic_optimization"
        )
    
    def get_next_id(self):
        """Get the next available strategy ID."""
        id_val = self.next_id
        self.next_id += 1
        return id_val
    
    def mutate(self, strategy):
        """
        Apply random mutations to a strategy.
        
        Args:
            strategy (StrategyParameters): Strategy to mutate
            
        Returns:
            StrategyParameters: Mutated strategy
        """
        # Create a copy with a new ID
        mutated = StrategyParameters(
            strategy_id=self.get_next_id(),
            sma_fast_period=strategy.sma_fast_period,
            sma_slow_period=strategy.sma_slow_period,
            rsi_period=strategy.rsi_period,
            rsi_buy_threshold=strategy.rsi_buy_threshold,
            rsi_sell_threshold=strategy.rsi_sell_threshold,
            rsi_buy_signal_threshold=strategy.rsi_buy_signal_threshold,
            rsi_sell_signal_threshold=strategy.rsi_sell_signal_threshold,
            volatility_period=strategy.volatility_period
        )
        
        # Apply random mutations
        if random.random() < self.mutation_rate:
            # SMA fast period (5-30)
            mutated.sma_fast_period = max(5, min(30, mutated.sma_fast_period + random.randint(-5, 5)))
            
        if random.random() < self.mutation_rate:
            # SMA slow period (must be > fast period)
            min_slow = mutated.sma_fast_period + 10
            mutated.sma_slow_period = max(min_slow, min(200, mutated.sma_slow_period + random.randint(-10, 10)))
            
        if random.random() < self.mutation_rate:
            # RSI period (7-21)
            mutated.rsi_period = max(7, min(21, mutated.rsi_period + random.randint(-2, 2)))
            
        if random.random() < self.mutation_rate:
            # RSI buy threshold (20-40)
            mutated.rsi_buy_threshold = max(20, min(40, mutated.rsi_buy_threshold + random.randint(-5, 5)))
            
        if random.random() < self.mutation_rate:
            # RSI sell threshold (60-80)
            mutated.rsi_sell_threshold = max(60, min(80, mutated.rsi_sell_threshold + random.randint(-5, 5)))
            
        if random.random() < self.mutation_rate:
            # RSI buy signal threshold (30-45)
            mutated.rsi_buy_signal_threshold = max(30, min(45, mutated.rsi_buy_signal_threshold + random.randint(-3, 3)))
            
        if random.random() < self.mutation_rate:
            # RSI sell signal threshold (55-70)
            mutated.rsi_sell_signal_threshold = max(55, min(70, mutated.rsi_sell_signal_threshold + random.randint(-3, 3)))
            
        if random.random() < self.mutation_rate:
            # Volatility period (10-30)
            mutated.volatility_period = max(10, min(30, mutated.volatility_period + random.randint(-3, 3)))
            
        return mutated
    
    def crossover(self, parent1, parent2):
        """
        Create a new strategy by crossing over two parent strategies.
        
        Args:
            parent1 (StrategyParameters): First parent strategy
            parent2 (StrategyParameters): Second parent strategy
            
        Returns:
            StrategyParameters: Child strategy
        """
        # Create a child with a new ID
        child = StrategyParameters(
            strategy_id=self.get_next_id()
        )
        
        # Randomly select parameters from parents
        if random.random() < 0.5:
            child.sma_fast_period = parent1.sma_fast_period
        else:
            child.sma_fast_period = parent2.sma_fast_period
            
        if random.random() < 0.5:
            child.sma_slow_period = parent1.sma_slow_period
        else:
            child.sma_slow_period = parent2.sma_slow_period
            
        # Ensure slow period > fast period
        if child.sma_slow_period <= child.sma_fast_period:
            child.sma_slow_period = child.sma_fast_period + 10
            
        if random.random() < 0.5:
            child.rsi_period = parent1.rsi_period
        else:
            child.rsi_period = parent2.rsi_period
            
        if random.random() < 0.5:
            child.rsi_buy_threshold = parent1.rsi_buy_threshold
        else:
            child.rsi_buy_threshold = parent2.rsi_buy_threshold
            
        if random.random() < 0.5:
            child.rsi_sell_threshold = parent1.rsi_sell_threshold
        else:
            child.rsi_sell_threshold = parent2.rsi_sell_threshold
            
        if random.random() < 0.5:
            child.rsi_buy_signal_threshold = parent1.rsi_buy_signal_threshold
        else:
            child.rsi_buy_signal_threshold = parent2.rsi_buy_signal_threshold
            
        if random.random() < 0.5:
            child.rsi_sell_signal_threshold = parent1.rsi_sell_signal_threshold
        else:
            child.rsi_sell_signal_threshold = parent2.rsi_sell_signal_threshold
            
        if random.random() < 0.5:
            child.volatility_period = parent1.volatility_period
        else:
            child.volatility_period = parent2.volatility_period
            
        return child
    
    def select_parents(self, population, fitness_scores):
        """
        Select parents for crossover using tournament selection.
        
        Args:
            population (list): List of StrategyParameters objects
            fitness_scores (list): Fitness scores for each strategy
            
        Returns:
            tuple: (parent1, parent2)
        """
        # Tournament selection
        tournament_size = 3
        
        # First parent
        idx1 = random.sample(range(len(population)), tournament_size)
        tournament1 = [(population[i], fitness_scores[i]) for i in idx1]
        parent1 = max(tournament1, key=lambda x: x[1])[0]
        
        # Second parent
        idx2 = random.sample(range(len(population)), tournament_size)
        tournament2 = [(population[i], fitness_scores[i]) for i in idx2]
        parent2 = max(tournament2, key=lambda x: x[1])[0]
        
        return parent1, parent2
    
    def evaluate_fitness(self, strategy):
        """
        Evaluate the fitness of a strategy by backtesting it.
        
        Args:
            strategy (StrategyParameters): Strategy to evaluate
            
        Returns:
            float: Fitness score
        """
        try:
            # Run the backtest
            _, metrics = backtest_parameterized_strategy(
                strategy,
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                db_manager=self.db_manager,
                logger=self.logger,
                plot_returns=False
            )
            
            # Save the metrics
            save_strategy_metrics(
                strategy.strategy_id,
                metrics,
                strategy.to_dict(),
                self.db_manager
            )
            
            # Return the specified fitness metric
            fitness = metrics.get(self.fitness_metric, 0)
            
            # Apply penalties for strategies with few trades
            if metrics['total_trades'] < 5:
                fitness *= 0.5  # Penalize strategies with few trades
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating strategy {strategy.strategy_id}: {str(e)}")
            return 0  # Assign zero fitness on error
    
    def evolve(self, initial_population=None):
        """
        Run the genetic evolution process.
        
        Args:
            initial_population (list, optional): Initial population of strategies
            
        Returns:
            list: The final population after evolution
        """
        start_time = time.time()
        
        # Start execution logging
        execution_id = self.logger.log_execution(
            execution_name=f"genetic_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Genetic optimization of trading strategies",
            parameters=json.dumps({
                "population_size": self.population_size,
                "generations": self.generations,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "elite_ratio": self.elite_ratio,
                "fitness_metric": self.fitness_metric
            }),
            script_id=self.script_id
        )
        
        print(f"Starting genetic optimization with {self.population_size} population size for {self.generations} generations")
        
        # Start with top strategies if initial population not provided
        if initial_population is None:
            # Get top strategies from the database
            top_strategies = get_top_strategies(
                num_strategies=10,  # Get top 10
                sort_by=self.fitness_metric,
                db_manager=self.db_manager
            )
            
            # Convert to StrategyParameters objects
            initial_population = []
            for _, row in top_strategies.iterrows():
                params = json.loads(row['parameters'])
                strategy = StrategyParameters(
                    strategy_id=params['strategy_id'],
                    sma_fast_period=params['sma_fast_period'],
                    sma_slow_period=params['sma_slow_period'],
                    rsi_period=params['rsi_period'],
                    rsi_buy_threshold=params['rsi_buy_threshold'],
                    rsi_sell_threshold=params['rsi_sell_threshold'],
                    rsi_buy_signal_threshold=params['rsi_buy_signal_threshold'],
                    rsi_sell_signal_threshold=params['rsi_sell_signal_threshold'],
                    volatility_period=params['volatility_period']
                )
                initial_population.append(strategy)
            
            # If we need more, add random strategies
            while len(initial_population) < self.population_size:
                strat_id = self.get_next_id()
                initial_population.append(
                    StrategyParameters(strategy_id=strat_id)
                )
        
        # Main evolution loop
        population = initial_population
        best_fitness = 0
        best_strategy = None
        
        for generation in range(self.generations):
            print(f"\nGeneration {generation+1}/{self.generations}")
            
            # Evaluate fitness for each strategy
            fitness_scores = []
            for i, strategy in enumerate(population):
                print(f"  Evaluating strategy {strategy.strategy_id} ({i+1}/{len(population)})...")
                fitness = self.evaluate_fitness(strategy)
                fitness_scores.append(fitness)
                print(f"    Fitness: {fitness:.4f}")
                
                # Update best strategy
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_strategy = strategy
            
            # Early stopping condition - if this is the last generation
            if generation == self.generations - 1:
                break
                
            # Sort by fitness
            population_fitness = list(zip(population, fitness_scores))
            population_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Print top performers
            print("\n  Top performers this generation:")
            for i in range(min(3, len(population_fitness))):
                strat, fit = population_fitness[i]
                print(f"    Strategy {strat.strategy_id}: Fitness = {fit:.4f}")
            
            # Select elites to carry forward
            new_population = [p for p, _ in population_fitness[:self.num_elites]]
            
            # Generate the rest of the population through crossover and mutation
            while len(new_population) < self.population_size:
                # Decide whether to do crossover
                if random.random() < self.crossover_rate and len(population) >= 2:
                    # Select parents and perform crossover
                    parent1, parent2 = self.select_parents(population, fitness_scores)
                    child = self.crossover(parent1, parent2)
                    
                    # Potentially mutate the child
                    if random.random() < self.mutation_rate:
                        child = self.mutate(child)
                        
                    new_population.append(child)
                else:
                    # Just mutate a random strategy
                    parent = random.choice(population)
                    mutated = self.mutate(parent)
                    new_population.append(mutated)
            
            # Update the population
            population = new_population
        
        # Generate a plot for the best strategy
        if best_strategy:
            print(f"\nBest strategy: {best_strategy.strategy_id} with fitness {best_fitness:.4f}")
            
            try:
                # Run one final backtest with plot generation
                backtest_parameterized_strategy(
                    best_strategy,
                    ticker=self.ticker,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    db_manager=self.db_manager,
                    logger=self.logger,
                    plot_returns=True
                )
            except Exception as e:
                print(f"Error generating plot for best strategy: {str(e)}")
        
        # Log successful completion
        self.logger.log_execution_end(
            execution_id,
            status="success",
            result_summary=f"Completed {self.generations} generations of evolution",
            result_metrics=json.dumps({
                "generations": self.generations,
                "population_size": self.population_size,
                "best_fitness": best_fitness,
                "best_strategy_id": best_strategy.strategy_id,
                "elapsed_time": time.time() - start_time
            })
        )
        
        return population

def run_genetic_optimization(ticker='SPY', 
                            population_size=50, 
                            generations=10, 
                            mutation_rate=0.2,
                            crossover_rate=0.7,
                            elite_ratio=0.1,
                            fitness_metric="sharpe_ratio"):
    """
    Run genetic optimization of trading strategies.
    
    Args:
        ticker (str): Ticker symbol to optimize for
        population_size (int): Size of the population in each generation
        generations (int): Number of generations to run
        mutation_rate (float): Probability of mutation (0-1)
        crossover_rate (float): Probability of crossover (0-1)
        elite_ratio (float): Ratio of top performers to keep (0-1)
        fitness_metric (str): Metric to use for fitness (sharpe_ratio, profit_factor)
        
    Returns:
        list: Final population of evolved strategies
    """
    try:
        # Initialize the optimizer
        optimizer = GeneticOptimizer(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_ratio=elite_ratio,
            fitness_metric=fitness_metric,
            ticker=ticker
        )
        
        # Run the evolution
        final_population = optimizer.evolve()
        
        return final_population
        
    except Exception as e:
        print(f"Error in genetic optimization: {str(e)}")
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimize trading strategies using genetic algorithms')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol to optimize for')
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--mutation-rate', type=float, default=0.2, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, default=0.7, help='Crossover rate')
    parser.add_argument('--elite-ratio', type=float, default=0.1, help='Elite ratio')
    parser.add_argument('--fitness', type=str, default='sharpe_ratio', 
                       choices=['sharpe_ratio', 'profit_factor', 'win_rate', 'annualized_return'],
                       help='Fitness metric')
    
    args = parser.parse_args()
    
    # Run the optimization
    run_genetic_optimization(
        ticker=args.ticker,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elite_ratio=args.elite_ratio,
        fitness_metric=args.fitness
    ) 