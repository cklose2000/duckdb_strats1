"""
Strategy Generator for Automated Backtesting

This module generates strategy variants with different parameters for automated backtesting.
Each strategy is defined by a set of parameters that will be used to create SQL queries.
"""

import random
import json
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.db_manager import DBManager
from utils.metadata_logger import MetadataLogger

class StrategyParameters:
    """Class representing a set of trading strategy parameters."""
    
    def __init__(self, 
                 strategy_id, 
                 sma_fast_period=20, 
                 sma_slow_period=50,
                 rsi_period=14,
                 rsi_buy_threshold=30,
                 rsi_sell_threshold=70,
                 rsi_buy_signal_threshold=40,
                 rsi_sell_signal_threshold=60,
                 volatility_period=20,
                 stop_loss_pct=0.02,
                 take_profit_pct=0.05,
                 max_holding_periods=48):
        """
        Initialize strategy parameters.
        
        Args:
            strategy_id (int): Unique ID for this strategy
            sma_fast_period (int): Fast SMA period
            sma_slow_period (int): Slow SMA period
            rsi_period (int): RSI calculation period
            rsi_buy_threshold (int): RSI threshold for buy signals
            rsi_sell_threshold (int): RSI threshold for sell signals
            rsi_buy_signal_threshold (int): Secondary RSI threshold for buys on signal days
            rsi_sell_signal_threshold (int): Secondary RSI threshold for sells on signal days
            volatility_period (int): Period for volatility calculation
            stop_loss_pct (float): Stop loss percentage (e.g., 0.02 for 2%)
            take_profit_pct (float): Take profit percentage (e.g., 0.05 for 5%)
            max_holding_periods (int): Maximum number of periods to hold a position
        """
        self.strategy_id = strategy_id
        self.sma_fast_period = sma_fast_period
        self.sma_slow_period = sma_slow_period
        self.rsi_period = rsi_period
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.rsi_buy_signal_threshold = rsi_buy_signal_threshold
        self.rsi_sell_signal_threshold = rsi_sell_signal_threshold
        self.volatility_period = volatility_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_periods = max_holding_periods
        
    def to_dict(self):
        """Convert parameters to a dictionary."""
        return {
            'strategy_id': self.strategy_id,
            'sma_fast_period': self.sma_fast_period,
            'sma_slow_period': self.sma_slow_period,
            'rsi_period': self.rsi_period,
            'rsi_buy_threshold': self.rsi_buy_threshold,
            'rsi_sell_threshold': self.rsi_sell_threshold,
            'rsi_buy_signal_threshold': self.rsi_buy_signal_threshold,
            'rsi_sell_signal_threshold': self.rsi_sell_signal_threshold,
            'volatility_period': self.volatility_period,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_holding_periods': self.max_holding_periods
        }
    
    def __str__(self):
        """String representation of the strategy parameters."""
        return f"Strategy {self.strategy_id}: SMA({self.sma_fast_period},{self.sma_slow_period}), RSI({self.rsi_period}), Buy({self.rsi_buy_threshold},{self.rsi_buy_signal_threshold}), Sell({self.rsi_sell_threshold},{self.rsi_sell_signal_threshold})"


def generate_random_strategy(strategy_id):
    """
    Generate a random strategy with reasonable parameters.
    
    Args:
        strategy_id (int): ID to assign to the strategy
        
    Returns:
        StrategyParameters: Strategy parameters object
    """
    # SMA period ranges - ensure fast period is smaller than slow period
    fast_period = random.randint(10, 30)
    slow_period = random.randint(fast_period + 5, fast_period + 30)
    
    # RSI parameters
    rsi_period = random.randint(5, 14)
    
    # Thresholds for RSI - ensure buy threshold is below sell threshold
    rsi_buy_threshold = random.randint(20, 40)
    rsi_sell_threshold = random.randint(60, 80)
    
    # Thresholds for signal days - ensure buy is below sell
    rsi_buy_signal_threshold = random.randint(30, 45)
    rsi_sell_signal_threshold = random.randint(55, 70)
    
    # Volatility period
    volatility_period = random.randint(10, 30)
    
    # Risk management parameters
    stop_loss_pct = round(random.uniform(0.01, 0.05), 3)  # 1% to 5% stop loss
    take_profit_pct = round(random.uniform(0.03, 0.10), 3)  # 3% to 10% take profit
    max_holding_periods = random.randint(12, 72)  # 12 to 72 periods (hourly data)
    
    return StrategyParameters(
        strategy_id=strategy_id,
        sma_fast_period=fast_period,
        sma_slow_period=slow_period,
        rsi_period=rsi_period,
        rsi_buy_threshold=rsi_buy_threshold,
        rsi_sell_threshold=rsi_sell_threshold,
        rsi_buy_signal_threshold=rsi_buy_signal_threshold,
        rsi_sell_signal_threshold=rsi_sell_signal_threshold,
        volatility_period=volatility_period,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_periods=max_holding_periods
    )

def generate_strategies(num_strategies=100):
    """
    Generate a specified number of random strategies.
    
    Args:
        num_strategies (int): Number of strategies to generate
        
    Returns:
        list: List of StrategyParameters objects
    """
    strategies = []
    for i in range(1, num_strategies + 1):
        strategies.append(generate_random_strategy(i))
    return strategies

def register_strategies_in_metadata(strategies, db_manager=None, logger=None):
    """
    Register generated strategies in the metadata system.
    
    Args:
        strategies (list): List of StrategyParameters objects
        db_manager (DBManager, optional): Database manager instance
        logger (MetadataLogger, optional): Logger instance
        
    Returns:
        list: List of model_ids registered in metadata
    """
    # Create DB manager and logger if not provided
    if db_manager is None:
        from utils.db_manager import DBManager
        db_manager = DBManager()
        db_manager.connect()
    
    if logger is None:
        from utils.metadata_logger import MetadataLogger
        logger = MetadataLogger(db_manager.conn)
    
    # Register each strategy in metadata
    model_ids = []
    
    for strategy in strategies:
        # Log the strategy as a model
        model_id = logger.log_model(
            model_name=f"Strategy_{strategy.strategy_id}",
            model_type="trading_strategy",
            description=f"Multi-timeframe strategy variant with parameters: {strategy}",
            parameters=strategy.to_dict(),
            features=["price", "volume", "SMA", "RSI"],
            performance_metrics=None,
            script_id=None
        )
        
        model_ids.append(model_id)
    
    return model_ids

if __name__ == "__main__":
    # Generate some sample strategies
    sample_strategies = generate_strategies(5)
    
    # Print them out
    for strat in sample_strategies:
        print(strat) 