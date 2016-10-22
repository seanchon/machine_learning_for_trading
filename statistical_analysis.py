from lib.get_data import get_data
from lib.plot import plot_data
from lib.time_function import how_long
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def test_run():
    # Read data
    df = get_data(['SPY', 'XOM', 'GOOG', 'GLD'], '2012-01-01', '2012-12-31')
    # plot_data(df)

    # Compute global statistics for each stock
    # print(df.mean())
    # print(df.median())
    # print(df.std())

    # Plot SPY data, retain matplotlib axis object
    ax = df['SPY'].plot(title="SPY rolling mean", label='SPY')

    # Compute rolling mean using a 20-day window
    rm_SPY = pd.Series(df['SPY']).rolling(window=20).mean()

    # Add rolling mean to same plot
    rm_SPY.plot(label='Rolling mean', ax=ax)

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    test_run()
