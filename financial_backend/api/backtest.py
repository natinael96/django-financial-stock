import pandas as pd
from .models import StockData
from .tasks import fetch_stock_data


def calculate_moving_average(prices, window):
    """
    Computes the moving average over a specified window (number of days) 
    for the provided price data.
    
    Args:
        prices (pd.Series): Series of stock prices.
        window (int): The number of days to calculate the moving average over.
    
    Returns:
        pd.Series: The calculated moving average.
    """
    return prices.rolling(window=window).mean()


def fetch_stock_data_if_needed(symbol):
    """
    Checks if stock data for the symbol is available in the database.
    If not, fetches it from an API and stores it.
    
    Args:
        symbol (str): Stock symbol to fetch data for.
    
    Returns:
        pd.DataFrame: Stock data for the symbol.
        str: An error message if something went wrong.
    """
    stock_data = StockData.objects.filter(symbol=symbol).order_by('timestamp')
    if not stock_data.exists():
        # Fetch from API and store in DB if data doesn't exist
        stock_data_df, error = fetch_stock_data(symbol=symbol, return_as_dataframe=True)
        if error:
            return None, error  # Handle error appropriately
    else:
        # Convert QuerySet to DataFrame
        stock_data_df = pd.DataFrame(list(stock_data.values('timestamp', 'close_price', 'open_price', 'high_price', 'low_price', 'volume')))
    return stock_data_df, None


def backtest_strategy(symbol, initial_investment, short_window=50, long_window=200):
    stock_data = StockData.objects.filter(symbol=symbol).order_by('timestamp')

    if not stock_data.exists():
        return {'status': 'error', 'message': f'No stock data available for {symbol}'}

    data = pd.DataFrame(list(stock_data.values('timestamp', 'close_price')))
    data.set_index('timestamp', inplace=True)

    # Calculate moving averages
    data['Short_MA'] = calculate_moving_average(data['close_price'], short_window)
    data['Long_MA'] = calculate_moving_average(data['close_price'], long_window)

    # Initialize variables for backtest
    cash = initial_investment
    shares = 0
    portfolio_value = initial_investment
    max_drawdown = 0
    peak_value = initial_investment
    trade_count = 0
    position_open = False  # Tracks if we have an open position

    for i in range(1, len(data)):
        close_price = float(data['close_price'].iloc[i])

        # Buy condition: Short MA crosses above Long MA
        if data['Short_MA'].iloc[i] > data['Long_MA'].iloc[i] and not position_open:
            shares = cash / close_price
            cash = 0
            position_open = True
            trade_count += 1

        # Sell condition: Short MA crosses below Long MA
        elif data['Short_MA'].iloc[i] < data['Long_MA'].iloc[i] and position_open:
            cash = shares * close_price
            shares = 0
            position_open = False
            trade_count += 1

        # Update portfolio value, max drawdown, and peak value
        portfolio_value = cash + shares * close_price
        peak_value = max(peak_value, portfolio_value)
        drawdown = (peak_value - portfolio_value) / peak_value
        max_drawdown = max(max_drawdown, drawdown)

    # If shares are still held, sell them at the last available price
    if shares > 0:
        cash = shares * float(data['close_price'].iloc[-1])
        shares = 0

    final_value = cash

    return {
        'Initial Investment': initial_investment,
        'Final Portfolio Value (USD)': final_value,
        'Total Return (%)': (final_value - initial_investment) / initial_investment * 100,
        'Max Drawdown (%)': max_drawdown * 100,
        'Number of Trades': trade_count,
        'actual_prices': list(data['close_price']),
        'dates': list(data.index)
    }



def calculate_metrics(backtest_results):
    """
    Derives key performance metrics from the results of the backtest.
    
    This function computes the Return on Investment (ROI) and the total number 
    of trades executed during the backtesting process.
    
    Args:
        backtest_results (dict): The result of the backtest which includes
        the initial investment, final portfolio value, and number of trades.
    
    Returns:
        dict: A dictionary containing calculated performance metrics.
    """
    initial_investment = backtest_results['Initial Investment']
    final_value = backtest_results["Final Portfolio Value (USD)"]
    roi = (final_value - initial_investment) / initial_investment * 100

    trade_count = backtest_results['Number of Trades']
    
    # Return calculated metrics including ROI and number of trades
    return {
        'ROI (%)': roi,
        'Total Trades': trade_count
    }
