
        -- Detailed trade analysis for a specific strategy
        SELECT 
            date,
            action,
            direction,
            price,
            entry_price,
            exit_price,
            trade_return,
            exit_reason
        FROM trade_log
        WHERE strategy_id = ? -- Replace with strategy ID
        ORDER BY date;

        -- Strategy performance comparison
        SELECT 
            strategy_id,
            total_trades,
            win_rate,
            profit_factor,
            sharpe_ratio,
            annualized_return,
            JSON_EXTRACT(parameters, '$.sma_fast_period') as fast_sma,
            JSON_EXTRACT(parameters, '$.sma_slow_period') as slow_sma,
            JSON_EXTRACT(parameters, '$.rsi_period') as rsi_period,
            JSON_EXTRACT(parameters, '$.rsi_buy_threshold') as rsi_buy,
            JSON_EXTRACT(parameters, '$.rsi_sell_threshold') as rsi_sell
        FROM strategy_analysis
        WHERE total_trades >= 5
        ORDER BY sharpe_ratio DESC;

        -- Performance by day of week
        SELECT 
            EXTRACT(DOW FROM date) as day_of_week,
            COUNT(*) as trade_count,
            ROUND(AVG(trade_return) * 100, 2) as avg_return,
            ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate
        FROM trade_log
        WHERE trade_return IS NOT NULL
        GROUP BY day_of_week
        ORDER BY day_of_week;

        -- Parameter correlation with performance
        WITH param_extract AS (
            SELECT 
                strategy_id,
                sharpe_ratio,
                profit_factor,
                CAST(JSON_EXTRACT(parameters, '$.sma_fast_period') AS INTEGER) as fast_sma,
                CAST(JSON_EXTRACT(parameters, '$.sma_slow_period') AS INTEGER) as slow_sma,
                CAST(JSON_EXTRACT(parameters, '$.rsi_period') AS INTEGER) as rsi_period,
                CAST(JSON_EXTRACT(parameters, '$.rsi_buy_threshold') AS INTEGER) as rsi_buy,
                CAST(JSON_EXTRACT(parameters, '$.rsi_sell_threshold') AS INTEGER) as rsi_sell,
                CAST(JSON_EXTRACT(parameters, '$.stop_loss_pct') AS FLOAT) as stop_loss,
                CAST(JSON_EXTRACT(parameters, '$.take_profit_pct') AS FLOAT) as take_profit
            FROM strategy_analysis
            WHERE total_trades >= 5
        )
        SELECT 
            CORR(sharpe_ratio, fast_sma) as corr_sharpe_fast_sma,
            CORR(sharpe_ratio, slow_sma) as corr_sharpe_slow_sma,
            CORR(sharpe_ratio, rsi_period) as corr_sharpe_rsi_period,
            CORR(sharpe_ratio, rsi_buy) as corr_sharpe_rsi_buy,
            CORR(sharpe_ratio, rsi_sell) as corr_sharpe_rsi_sell,
            CORR(sharpe_ratio, stop_loss) as corr_sharpe_stop_loss,
            CORR(sharpe_ratio, take_profit) as corr_sharpe_take_profit
        FROM param_extract;
        