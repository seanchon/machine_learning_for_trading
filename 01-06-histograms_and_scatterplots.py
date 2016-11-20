"""Plot a histogram."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    symbols = ['SPY', 'XOM']
    df = get_data(symbols, '2009-01-01', '2012-12-31')
    plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily Returns")

    # Plot a histogram
    for symbol in symbols:
        daily_returns[symbol].hist(bins=20, label=symbol)

    # Get mean and standard deviation
    for symbol in symbols:
        mean = daily_returns[symbol].mean()
        print("{}: mean={}".format(symbol, mean))
        std = daily_returns[symbol].std()
        print("{}: std={}".format(symbol, std))

    # Compute kurtosis
    print(daily_returns.kurtosis())

    # Add mean and std to graph
    # plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    # plt.axvline(mean + std, color='r', linestyle='dashed', linewidth=2)
    # plt.axvline(mean - std, color='r', linestyle='dashed', linewidth=2)

    plt.legend(loc='upper right')
    plt.show()

def test_run2():
    # Read data
    symbols = ['SPY', 'XOM', 'GOLD']
    df = get_data(symbols, '2009-01-01', '2012-12-31')

    # Compute daily returns
    daily_returns = compute_daily_returns(df)

    # Scatterplot SPY vs XOM
    daily_returns.plot(kind='scatter', x='SPY', y='XOM')
    beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1)
    plt.plot(daily_returns['SPY'], beta_XOM * daily_returns['SPY'] + alpha_XOM, '-', color='r')
    plt.show()

    # Scatterplot SPY vs XOM
    daily_returns.plot(kind='scatter', x='SPY', y='GOLD')
    beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['GOLD'], 1)
    plt.plot(daily_returns['SPY'], beta_XOM * daily_returns['SPY'] + alpha_XOM, '-', color='r')
    plt.show()


if __name__ == '__main__':
    test_run2()
