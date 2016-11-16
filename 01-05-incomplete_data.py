import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# function to get path of the symbol
def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


# reads CSV
def get_data(symbol_list, dates):
    df_final = pd.DataFrame(index=dates)
    if "SPY" not in symbol_list:
        symbol_list.insert(0, "SPY")
    for symbol in symbol_list:
        file_path = symbol_to_path(symbol)
        df_temp = pd.read_csv(file_path, parse_dates=True, index_col="Date", usecols=["Date", "Adj Close"])  # check the rest of this line
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df_final = df_final.join(df_temp)
        if symbol == "SPY":
            df_final = df_final.dropna(subset=['SPY'])

    return df_final


# plot function
def plot(df_data):
    ax = df_data.plot(title="Incomplete Data", fontsize=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


if __name__ == '__main__':
    # list of symbols
    # symbol_list = ["PSX", "FAKE1", "FAKE2"]
    symbol_list = ["AAPL"]

    # date range
    start_date = '2015-01-01'
    end_date = '2015-12-31'

    # create date range
    idx = pd.date_range(start_date, end_date)

    # get adjusted close of each symbol
    df_data = get_data(symbol_list, idx)

    # forward fill missing data FIRST
    df_data.fillna(method="ffill", inplace="TRUE")
    # then back fill missing data NEXT
    df_data.fillna(method='bfill', inplace="TRUE")

    plot(df_data)
