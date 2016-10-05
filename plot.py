from lib.get_data import get_stock_csv
from lib.plot import plot_data
import matplotlib.pyplot as plt
import pandas as pd


def test_run():
    # Define date range
    start_date = '2015-01-01'
    end_date = '2015-12-31'
    dates = pd.date_range(start_date, end_date)

    # Create an empty dataframe
    df = pd.DataFrame(index=dates)

    # Read SPY data into temporary DataFrame
    symbol = 'SPY'
    get_stock_csv(symbol, start_date, end_date, "data/{}.csv".format(symbol))
    dfSPY = pd.read_csv("data/SPY.csv", index_col="Date", parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])

    # Rename 'Adj Close' column to 'SPY' to prevent clash
    dfSPY = dfSPY.rename(columns={'Adj Close': 'SPY'})

    # Join the two dataframes using DataFrame.join()
    # df = df.join(dfSPY)

    # Drop NaN Values
    # df = df.dropna()

    # Combine the above two steps
    df = df.join(dfSPY, how="inner")

    # Read in more stocks
    symbols = ['GOOG', 'IBM', 'GLD']
    for symbol in symbols:
        # populate dataframe
        get_stock_csv(symbol, start_date, end_date, "data/{}.csv".format(symbol))
        df_temp = pd.read_csv("data/{}.csv".format(symbol), index_col="Date", parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)

    return df

if __name__ == '__main__':
    df = test_run()
    plot_data(df)
    # normalize dataframe
    df = df / df.ix[0]
    # slicing a dataframe. df.ix is considered more pythonic
    df.ix['2010-01-01':'2010-12-31', ['SPY', 'GOOG', 'IBM', 'GLD']].plot()
    plt.show()
