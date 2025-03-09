"""
Analysis utilities for the DuckDB backtesting framework.

This package contains scripts for viewing and analyzing backtest results,
including trade analysis, performance metrics, and visualization tools.
"""

from utils.analysis.view_trades import view_trades
from utils.analysis.view_backtest_details import analyze_backtest_details

__all__ = ['view_trades', 'analyze_backtest_details'] 