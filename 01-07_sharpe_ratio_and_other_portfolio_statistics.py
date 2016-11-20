import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.get_data import get_data
from lib.plot import plot_data

"""
Check that the following equations are correct.
It would be beneficial to write a couple of tests to make sure that some of the following calculations
can be applied to the original data to get correct starting and ending values.
"""

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.ix[0, :] = 0  # set daily returns for row 0 to 0
    return daily_returns


def compute_cumulative_returns(df):
    cumulative_returns = {}
    for symbol in df.keys():
        cumulative_returns[symbol] = ((df[symbol][-1] - df[symbol][0]) / df[symbol][0])

    return cumulative_returns


def compute_average_daily_returns(df):
    daily_returns = compute_daily_returns(df)
    average_daily_returns = {}
    for symbol in daily_returns.keys():
        average_daily_returns[symbol] = daily_returns[symbol].mean()

    return average_daily_returns


def compute_standard_deviations_of_daily_returns(df):
    daily_returns = compute_daily_returns(df)
    standard_deviations = {}
    for symbol in daily_returns.keys():
        standard_deviations[symbol] = daily_returns[symbol].std()

    return standard_deviations


def compute_sharpe_ratios(df, trading_days=252, daily_risk_free_rate=0):
    average_daily_returns = compute_average_daily_returns(df)
    standard_deviations = compute_standard_deviations_of_daily_returns(df)

    sharpe_ratios = {}
    for symbol in average_daily_returns.keys():
        sharpe_ratios[symbol] = math.sqrt(trading_days) * ((average_daily_returns[symbol] - daily_risk_free_rate) / standard_deviations[symbol])

    return sharpe_ratios


if __name__ == '__main__':
    symbols = ['SPY', 'GOOG', 'AAPL', 'XOM']
    start_date = '2015-01-01'
    end_date = '2015-12-31'

    df = get_data(symbols, start_date, end_date)
    print("DAILY RETURNS")
    print(compute_daily_returns(df))
    print("CUMULATIVE RETURNS")
    print(compute_cumulative_returns(df))
    print("AVERAGE DAILY RETURNS")
    print(compute_average_daily_returns(df))
    print("STANDARD DEVIATIONS OF DAILY RETURNS")
    print(compute_standard_deviations_of_daily_returns(df))
    print("SHARPE RATIOS")
    print(compute_sharpe_ratios(df))
