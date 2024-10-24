import numpy as np
import pandas as pd
from api.models import StockData

def movingAverageBacktest(symbol, short_window=50, long_window=200, initial_investment=10000):
    data = StockData.objects.filter(symbol=symbol).order_by('date')
    df = pd.DataFrame(list(data.values('date', 'closePrice')))
    df['closePrice'] = pd.to_numeric(df['closePrice'])
    df.set_index('date', inplace=True)

    # Calculate moving averages
    df['shortMAvg'] = df['closePrice'].rolling(window=short_window, min_periods=1).mean()
    df['longMAvg'] = df['closePrice'].rolling(window=long_window, min_periods=1).mean()

    # Generate trading signals
    df['signal'] = 0
    df['signal'][short_window:] = np.where(df['shortMAvg'][short_window:] > df['longMAvg'][short_window:], 1, 0)
    df['positions'] = df['signal'].diff()

    # Simulate backtest
    positions = initial_investment * df['signal']
    df['portfolioVal'] = positions + initial_investment

    return df
