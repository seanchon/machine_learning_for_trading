import get_data
import matplotlib.pyplot as plt
import pandas as pd


def get_rolling_mean(df, window=20):
    return pd.Series(df).rolling(window=window).mean()


def get_rolling_std(df, window=20):
    return pd.Series(df).rolling(window=window).std()


def get_bollinger_bands(rm, rstd, num_rstd=2):
    return rm + num_rstd * rstd, rm - num_rstd * rstd


def plot_bollinger_band(stock_symbol, start_date, end_date, window=20):
    df = get_data.get_data([stock_symbol], start_date, end_date)

    # Compute Bollinger Bands
    # 1. Compute rolling mean using a 20-day window
    rm = get_rolling_mean(df[stock_symbol], window=window)

    # 2. Compute rolling standard deviation
    rstd = get_rolling_std(df[stock_symbol], window=window)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)

    # Plot raw SPY data, rolling mean and Bollinger Bands
    ax = df[stock_symbol].plot(title="Bollinger Bands", label=stock_symbol)
    rm.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()
