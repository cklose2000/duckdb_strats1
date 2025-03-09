
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
            