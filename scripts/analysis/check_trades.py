import duckdb
import sys

def check_trades(strategy_id):
    # Connect to the database
    con = duckdb.connect('db/backtest.ddb')
    
    # Get strategy metrics
    metrics = con.execute(
        f"SELECT * FROM strategy_performance_summary WHERE strategy_id = {strategy_id}"
    ).fetchone()
    
    if not metrics:
        print(f"No metrics found for strategy {strategy_id}")
        return
    
    print(f"Strategy {strategy_id} metrics:")
    print(f"  Total trades: {metrics[1]}")
    print(f"  Win rate: {metrics[2]}%")
    print(f"  Profit factor: {metrics[3]}")
    print(f"  Average return: {metrics[4]:.2f}%")
    print(f"  Max drawdown: {metrics[5]:.2f}%")
    print(f"  Sharpe ratio: {metrics[6]:.2f}")
    print(f"  Annualized return: {metrics[7]:.2f}%")
    print(f"  Parameters: {metrics[8]}")
    
    # Get trades
    trades = con.execute(
        f"""
        SELECT 
            date, 
            close, 
            position, 
            trade_action, 
            entry_price, 
            exit_price, 
            trade_return, 
            exit_reason 
        FROM parameterized_backtest_results 
        WHERE strategy_id = {strategy_id} 
        AND trade_action IS NOT NULL 
        ORDER BY date
        """
    ).fetchall()
    
    print("\nTrades:")
    for trade in trades:
        date, close, position, action, entry, exit, ret, reason = trade
        ret_str = f"{ret:.4f}" if ret is not None else "None"
        print(f"  {date} - {action} - Position: {position} - Entry: {entry} - Exit: {exit} - Return: {ret_str} - Reason: {reason}")
    
    # Close the connection
    con.close()

if __name__ == "__main__":
    strategy_id = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    check_trades(strategy_id) 