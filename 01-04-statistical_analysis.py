from lib.get_data import get_data
from lib.plot import plot_data
from lib.time_function import how_long
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()  # copy given DataFrame to match size and column names
    # compute daily returns for row 1 onwards
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.ix[0] = 0  # set daily returns for row 0 to 0
    return daily_returns


def compute_daily_returns_2(df):
    """Compute and return the daily return values."""
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.ix[0] = 0
    return daily_returns


def compute_cumulative_returns(df):
    return compute_daily_returns(df).cumsum()


def get_rolling_mean(df, window=20):
    return pd.Series(df).rolling(window=window).mean()


def get_rolling_std(df, window=20):
    return pd.Series(df).rolling(window=window).std()


def get_bollinger_bands(rm, rstd):
    return rm + 2 * rstd, rm - 2 * rstd


def test_run():
    stock_symbol = 'EME'
    # Read data
    df = get_data(['SPY', stock_symbol], '2016-01-01', '2016-10-24')
    # plot_data(df)

    # Compute global statistics for each stock
    # print(df.mean())
    # print(df.median())
    # print(df.std())

    # Compute Bollinger Bands
    # 1. Compute rolling mean using a 20-day window
    rm_SPY = get_rolling_mean(df[stock_symbol], window=20)

    # 2. Compute rolling standard deviation
    rstd_SPY = get_rolling_std(df[stock_symbol], window=20)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)

    # Plot raw SPY data, rolling mean and Bollinger Bands
    ax = df[stock_symbol].plot(title="Bollinger Bands", label=stock_symbol)
    rm_SPY.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    test_run()
