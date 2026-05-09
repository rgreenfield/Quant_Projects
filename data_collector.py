import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timezone


def _download_yahoo_chart(symbol, start_date, end_date):
    """Download Yahoo Finance chart data directly when yfinance fails."""
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "includeAdjustedClose": "true",
    }
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo chart API returned no results for '{symbol}'.")
    result = result[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    quote = indicators.get("quote", [{}])[0]
    adjclose = indicators.get("adjclose", [{}])[0].get("adjclose")
    if not timestamps:
        raise RuntimeError(f"Yahoo chart API returned no price timestamps for '{symbol}'.")
    df = pd.DataFrame(
        {
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Adj Close": adjclose if adjclose is not None else quote.get("close", []),
            "Volume": quote.get("volume", []),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True),
    )
    df.index.name = "Date"
    return df


def download_stock_data(symbol, start_date, end_date):
    """Download historical stock data for a given symbol and date range.

    Args:
        symbol (str): Stock ticker symbol.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pandas.DataFrame: Historical OHLCV stock data.

    Raises:
        RuntimeError: If the symbol data cannot be downloaded.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, threads=False)
    except Exception as exc:
        print(f"Failed to download '{symbol}' using yf.download: {exc}")
        data = pd.DataFrame()

    if data.empty:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1d', actions=False)
        except Exception as exc:
            print(f"Failed to download '{symbol}' using yf.Ticker.history: {exc}")
            data = pd.DataFrame()

    if data.empty:
        try:
            print(f"Attempting direct Yahoo chart download for '{symbol}'...")
            data = _download_yahoo_chart(symbol, start_date, end_date)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download '{symbol}' from Yahoo Finance: {exc}"
            ) from exc

    if data.empty:
        raise RuntimeError(
            f"No historical data downloaded for '{symbol}'. "
            "Check the ticker symbol, date range, and network connectivity."
        )

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








