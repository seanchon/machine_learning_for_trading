from lib.get_data import get_stock_csv as get_stock_csv
import matplotlib.pyplot as plt
import pandas as pd


def normalize_data(df):
    return df / df.ix[0]


def plot_data(df, title="Stock Prices", normalize=False):
    '''Plot Stock Prices'''
    if normalize:
        df = normalize_data(df)
    ax = df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def plot_selected_data(df, columns, start_index, end_index, normalize=False):
    if normalize:
        df = df / df.ix[0]
    plot_data(df.ix[start_index:end_index, columns], title="Selected Data", normalize=normalize)


def plot_single_stock(symbol, start_date, end_date):
    filename = "data/{}.csv".format(symbol)
    get_stock_csv(symbol, start_date, end_date, filename)

    df = pd.read_csv("data/{}.csv".format(symbol))
    df[['Close', 'Adj Close']].plot()

    plt.show()


def create_multiple_stock_dataframe(symbol_list, start_date, end_date):
    # Define date range
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
    for symbol in symbol_list:
        # populate dataframe
        get_stock_csv(symbol, start_date, end_date, "data/{}.csv".format(symbol))
        df_temp = pd.read_csv("data/{}.csv".format(symbol), index_col="Date", parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)

    return df
