-- Multi-Timeframe Analysis Query
-- This query analyzes price data across different timeframes
-- to identify trading opportunities and patterns

-- Parameters:
--   ticker: Stock ticker to analyze (e.g., 'SPY')
--   start_date: Start date for analysis (e.g., '2024-01-01')
--   end_date: End date for analysis (e.g., '2025-02-28')

-- First, get daily price data with momentum indicators
WITH daily_indicators AS (
    SELECT
        ticker,
        date,
        open,
        high,
        low,
        close,
        volume,
        
        -- Price movement
        close / LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return,
        close / LAG(close, 5) OVER (PARTITION BY ticker ORDER BY date) - 1 AS weekly_return,
        close / LAG(close, 20) OVER (PARTITION BY ticker ORDER BY date) - 1 AS monthly_return,
        
        -- Simple Moving Averages
        AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS sma_10,
        AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
        AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma_50,
        AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS sma_200,
        
        -- Volatility
        (high - low) / close AS daily_range,
        STDDEV(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) / 
        AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volatility_20,
        
        -- Volume analysis
        volume / AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS relative_volume
    FROM price_data_daily
    WHERE ticker = ? -- Parameter 1: ticker
    AND date BETWEEN ? AND ? -- Parameters 2 & 3: start_date and end_date
),

-- Get hourly data for intraday patterns
hourly_indicators AS (
    SELECT 
        ticker,
        date,
        close,
        -- Hourly momentum
        close / LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) - 1 AS hourly_return,
        -- Hourly volatility
        (high - low) / close AS hourly_range,
        -- Session indicators (time of day patterns)
        EXTRACT(HOUR FROM date) AS hour_of_day,
        -- Volume pattern
        volume / AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS hourly_relative_volume
    FROM price_data_60min
    WHERE ticker = ? -- Parameter 1 again: ticker
    AND date BETWEEN ? AND ? -- Parameters 2 & 3 again: start_date and end_date
),

-- Get 5-minute data for short-term trading signals
minute_indicators AS (
    SELECT
        ticker,
        date,
        close,
        -- 5-min momentum
        close / LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) - 1 AS minute_return,
        -- Short-term volatility
        (high - low) / close AS minute_range,
        -- Time markers
        EXTRACT(HOUR FROM date) AS hour,
        EXTRACT(MINUTE FROM date) AS minute,
        EXTRACT(DOW FROM date) AS day_of_week,
        -- Volume spikes
        CASE WHEN volume > 2 * AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) 
             THEN 1 ELSE 0 END AS volume_spike
    FROM price_data_5min
    WHERE ticker = ? -- Parameter 1 again: ticker
    AND date BETWEEN ? AND ? -- Parameters 2 & 3 again: start_date and end_date
),

-- Combine daily data with future returns for backtesting
backtest_data AS (
    SELECT
        d.ticker,
        d.date,
        d.close,
        d.daily_return,
        d.weekly_return,
        d.monthly_return,
        d.sma_10,
        d.sma_20,
        d.sma_50,
        d.sma_200,
        d.daily_range,
        d.volatility_20,
        d.relative_volume,
        
        -- Price relative to moving averages
        d.close / d.sma_10 - 1 AS price_rel_sma_10,
        d.close / d.sma_20 - 1 AS price_rel_sma_20,
        d.close / d.sma_50 - 1 AS price_rel_sma_50,
        d.close / d.sma_200 - 1 AS price_rel_sma_200,
        
        -- Simple signals
        CASE WHEN d.close > d.sma_20 THEN 1 ELSE 0 END AS above_sma_20,
        CASE WHEN d.sma_10 > d.sma_20 THEN 1 ELSE 0 END AS sma_10_above_20,
        CASE WHEN d.close > d.sma_200 THEN 1 ELSE 0 END AS above_sma_200,
        
        -- Future returns for backtesting
        LEAD(d.close, 1) OVER (PARTITION BY d.ticker ORDER BY d.date) / d.close - 1 AS next_day_return,
        LEAD(d.close, 5) OVER (PARTITION BY d.ticker ORDER BY d.date) / d.close - 1 AS next_week_return,
        LEAD(d.close, 20) OVER (PARTITION BY d.ticker ORDER BY d.date) / d.close - 1 AS next_month_return
    FROM daily_indicators d
),

-- Daily session analysis (for day trading strategies)
daily_session AS (
    SELECT
        CAST(date AS DATE) AS trade_date,
        ticker,
        MIN(close) AS session_low,
        MAX(close) AS session_high,
        (MAX(close) - MIN(close)) / MIN(close) AS session_range,
        SUM(volume) AS session_volume,
        COUNT(*) AS bar_count
    FROM price_data_5min
    WHERE ticker = ? -- Parameter 1 again: ticker
    AND date BETWEEN ? AND ? -- Parameters 2 & 3 again: start_date and end_date
    GROUP BY CAST(date AS DATE), ticker
),

-- Multi-timeframe potential signals
multi_timeframe_signals AS (
    SELECT
        CAST(d.date AS DATE) AS signal_date,
        d.ticker,
        d.close AS daily_close,
        d.price_rel_sma_20,
        d.price_rel_sma_50,
        d.price_rel_sma_200,
        d.volatility_20,
        
        -- Combine multi-timeframe data
        h.hourly_range,
        m.minute_range,
        s.session_range,
        
        -- Signal generation
        CASE 
            WHEN d.price_rel_sma_20 > 0 AND d.price_rel_sma_50 > 0 AND d.volatility_20 < 0.02 THEN 'Strong_Uptrend'
            WHEN d.price_rel_sma_20 < 0 AND d.price_rel_sma_50 < 0 AND d.volatility_20 < 0.02 THEN 'Strong_Downtrend'
            WHEN d.price_rel_sma_20 > 0 AND d.price_rel_sma_50 < 0 THEN 'Mixed_Trend'
            WHEN d.volatility_20 > 0.03 THEN 'High_Volatility'
            ELSE 'Neutral'
        END AS trend_signal,
        
        -- Combine timeframe momentum
        d.daily_return,
        AVG(h.hourly_return) AS avg_hourly_return,
        AVG(m.minute_return) AS avg_minute_return,
        
        -- Future returns for signal testing
        d.next_day_return,
        d.next_week_return,
        d.next_month_return
    FROM backtest_data d
    LEFT JOIN daily_session s ON CAST(d.date AS DATE) = s.trade_date AND d.ticker = s.ticker
    LEFT JOIN hourly_indicators h ON CAST(d.date AS DATE) = CAST(h.date AS DATE) AND d.ticker = h.ticker
    LEFT JOIN minute_indicators m ON CAST(d.date AS DATE) = CAST(m.date AS DATE) AND d.ticker = m.ticker
    GROUP BY 
        CAST(d.date AS DATE),
        d.ticker,
        d.close,
        d.price_rel_sma_20,
        d.price_rel_sma_50,
        d.price_rel_sma_200,
        d.volatility_20,
        d.daily_return,
        s.session_range,
        d.next_day_return,
        d.next_week_return,
        d.next_month_return
)

-- Main query output with signal performance
SELECT
    signal_date,
    ticker,
    daily_close,
    trend_signal,
    
    -- Volatility comparison
    volatility_20 AS daily_volatility,
    hourly_range,
    minute_range,
    session_range,
    
    -- Momentum across timeframes
    daily_return * 100 AS daily_return_pct,
    avg_hourly_return * 100 AS hourly_return_pct,
    avg_minute_return * 100 AS minute_return_pct,
    
    -- Future performance
    next_day_return * 100 AS next_day_return_pct,
    next_week_return * 100 AS next_week_return_pct,
    next_month_return * 100 AS next_month_return_pct,
    
    -- Signal effectiveness
    CASE WHEN trend_signal = 'Strong_Uptrend' AND next_day_return > 0 THEN 1
         WHEN trend_signal = 'Strong_Downtrend' AND next_day_return < 0 THEN 1
         ELSE 0 END AS signal_accuracy,
         
    -- Position sizing recommendation based on volatility
    CASE
        WHEN volatility_20 < 0.01 THEN 'Large'
        WHEN volatility_20 < 0.02 THEN 'Medium'
        WHEN volatility_20 < 0.03 THEN 'Small'
        ELSE 'Avoid'
    END AS position_size_recommendation
FROM multi_timeframe_signals
ORDER BY signal_date; 