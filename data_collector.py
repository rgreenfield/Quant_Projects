import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def download_stock_data(symbol, start_date, end_date):
    """Download historical stock data for a given symbol and date range.

    Args:
        symbol (str): Stock ticker symbol.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pandas.DataFrame: Historical OHLCV stock data.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data


def clean_data(df):
    """Clean stock data by dropping missing rows, removing duplicate dates, and checking prices.

    Args:
        df (pandas.DataFrame): Raw stock data with a DatetimeIndex.

    Returns:
        pandas.DataFrame: Cleaned stock data.
    """
    cleaned = df.dropna()
    cleaned = cleaned[~cleaned.index.duplicated()]

    if not cleaned[['Open', 'High', 'Low', 'Close']].gt(0).all().all():
        print("Warning: some price values are not positive.")

    return cleaned


def add_returns(df):
    """Add percentage and log returns to the DataFrame.

    Args:
        df (pandas.DataFrame): Stock data with a Close column.

    Returns:
        pandas.DataFrame: Data with daily_return and log_return columns added.
    """
    df = df.copy()
    df['daily_return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df


def save_data(df, filepath):
    """Save the DataFrame to CSV."""
    df.to_csv(filepath)


def load_data(filepath):
    """Load a DataFrame from CSV and parse the index as dates."""
    return pd.read_csv(filepath, index_col=0, parse_dates=[0])


def plot_price_and_returns(df, symbol):
    """Plot closing price and daily returns for a symbol."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df.index, df['Close'], label='Close')
    axes[0].set_title(f'{symbol} Closing Price')
    axes[0].set_ylabel('Price')
    axes[0].grid(True)

    axes[1].plot(df.index, df['daily_return'], label='Daily Return', color='orange')
    axes[1].set_title(f'{symbol} Daily Returns')
    axes[1].set_ylabel('Return')
    axes[1].set_xlabel('Date')
    axes[1].grid(True)

    plt.tight_layout()
    return fig








