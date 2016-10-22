from lib.get_data import get_data
from lib.plot import plot_data
from lib.time_function import how_long
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def get_rolling_mean(df, window=20):
    return pd.Series(df).rolling(window=window).mean()


def get_rolling_std(df, window=20):
    return pd.Series(df).rolling(window=window).std()


def get_bollinger_bands(rm, rstd):
    return rm + 2 * rstd, rm - 2 * rstd


def test_run():
    # Read data
    df = get_data(['SPY', 'XOM', 'GOOG', 'GLD'], '2012-01-01', '2012-12-31')
    # plot_data(df)

    # Compute global statistics for each stock
    # print(df.mean())
    # print(df.median())
    # print(df.std())

    # Compute Bollinger Bands
    # 1. Compute rolling mean using a 20-day window
    rm_SPY = get_rolling_mean(df['SPY'], window=20)

    # 2. Compute rolling standard deviation
    rstd_SPY = get_rolling_std(df['SPY'], window=20)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)

    # Plot raw SPY data, rolling mean and Bollinger Bands
    ax = df['SPY'].plot(title="Bollinger Bands", label='SPY')
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
