import pandas as pd

def add_moving_averages(df: pd.DataFrame, windows=[5, 10, 20, 50]) -> pd.DataFrame:
    """Add simple moving averages for closing prices.

    Args:
        df (pd.DataFrame): DataFrame with a closing price column named 'Close'.
        windows (list[int], optional): Window sizes for SMAs. Defaults to [5, 10, 20, 50].

    Returns:
        pd.DataFrame: DataFrame with SMA columns added.

    A 5-day SMA smooths out daily noise and shows the short-term trend.
    """
    df = df.copy()
    for w in windows:
        df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add the Relative Strength Index (RSI) to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a closing price column named 'Close'.
        period (int, optional): Lookback period for RSI. Defaults to 14.

    Returns:
        pd.DataFrame: DataFrame with an 'RSI' column added.

    RSI measures whether a stock is overbought or oversold.
    RSI above 70 is typically overbought; below 30 is oversold.
    """
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD and signal line columns to the DataFrame.

    MACD shows momentum shifts by comparing fast and slow EMAs.
    """
    df = df.copy()
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add Bollinger Bands and position to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a closing price column named 'Close'.
        window (int, optional): Rolling window for the middle SMA and std. Defaults to 20.

    Returns:
        pd.DataFrame: DataFrame with Bollinger band columns added.

    The position indicates where the current close sits within the bands.
    """
    df = df.copy()
    middle = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    df['BB_middle'] = middle
    df['BB_upper'] = middle + 2 * std
    df['BB_lower'] = middle - 2 * std
    df['bb_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a volume column named 'Volume'.

    Returns:
        pd.DataFrame: DataFrame with volume SMA and ratio columns added.

    High volume ratio often signals important price moves.
    """
    df = df.copy()
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    return df


def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Add a future return target to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a closing price column named 'Close'.
        horizon (int, optional): Number of periods ahead for the target. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with a 'target' column added.

    This uses shift(-horizon) to look into the future only for the target, never for features.
    """
    df = df.copy()
    df['target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df


def prepare_all_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Prepare all features and the target, then drop NaN rows.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' and 'Volume' columns.
        horizon (int, optional): Target horizon in periods. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with all feature columns and target, NaNs removed.
    """
    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_features(df)
    df = add_target(df, horizon=horizon)
    df = df.dropna()
    print('Prepared features:', df.columns.tolist())
    print('Shape:', df.shape)
    return df



