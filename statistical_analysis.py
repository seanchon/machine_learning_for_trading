from lib.get_data import get_data
from lib.plot import plot_data
from lib.time_function import how_long
import numpy as np
import time


def test_run():
    # Read data
    df = get_data(['SPY', 'XOM', 'GOOG', 'GLD'], '2010-01-01', '2012-12-31')
    plot_data(df)

    # Compute global statistics for each stock
    print(df.mean())
    print(df.median())
    print(df.std())


if __name__ == "__main__":
    test_run()
