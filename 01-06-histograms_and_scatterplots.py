"""Plot a histogram."""

import pandas as pd
import matplotlib.pyplot as plt

from lib.get_data import get_data
from lib.plot import plot_data

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.ix[0, :] = 0  # set daily returns for row 0 to 0
    return daily_returns


def test_run():
    # Read data
    symbols = ['SPY']
    df = get_data(symbols, '2009-01-01', '2012-12-31')
    plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily Returns")

    # Plot a histogram
    daily_returns.hist(bins=20)
    plt.show()


if __name__ == '__main__':
    test_run()
