import pandas as pd


def simple_split(df: pd.DataFrame, train_pct: float = 0.7, val_pct: float = 0.15) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split DataFrame chronologically into train, validation, and test sets with features and target separated.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        train_pct (float, optional): Percentage for training set. Defaults to 0.7.
        val_pct (float, optional): Percentage for validation set. Defaults to 0.15.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: 
            X_train, y_train, X_val, y_val, X_test, y_test.

    Splits are chronological (no shuffling) based on row order.
    Features exclude 'target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close'.
    Target is 'target'.
    """
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Separate features and target for each split
    exclude_cols = ['target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
    X_train = train_df.drop(columns=exclude_cols)
    y_train = train_df['target']
    X_val = val_df.drop(columns=exclude_cols)
    y_val = val_df['target']
    X_test = test_df.drop(columns=exclude_cols)
    y_test = test_df['target']

    return X_train, y_train, X_val, y_val, X_test, y_test


def walk_forward_split(df: pd.DataFrame, train_size: int = 252, test_size: int = 21):
    """Generator for walk-forward validation splits.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        train_size (int, optional): Initial training window size. Defaults to 252.
        test_size (int, optional): Test window size. Defaults to 21.

    Yields:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) pairs.

    Starts with train 0 to train_size-1, test train_size to train_size+test_size-1.
    Then expands train by test_size, slides test forward by test_size.
    """
    n = len(df)
    start = 0
    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size
        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]
        yield train_df, test_df
        start += test_size


def validate_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Validate that there is no data leakage between train and test sets.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Test DataFrame.

    Raises:
        ValueError: If train_df.index.max() >= test_df.index.min(), indicating potential leakage.
    """
    if train_df.index.max() >= test_df.index.min():
        raise ValueError(
            f"Data leakage detected: train max index {train_df.index.max()} >= test min index {test_df.index.min()}. "
            "Ensure chronological splits to prevent future data leaking into training."
        )


def print_split_info(name: str, df: pd.DataFrame) -> None:
    """Print information about a split DataFrame.

    Args:
        name (str): Name of the split (e.g., 'Train', 'Validation', 'Test').
        df (pd.DataFrame): The DataFrame to describe.
    """
    if df.empty:
        print(f"{name}: Empty DataFrame")
        return
    first_date = df.index.min()
    last_date = df.index.max()
    num_samples = len(df)
    print(f"{name}: {num_samples} samples from {first_date.date()} to {last_date.date()}")


def test_splits():
    """Test the splitting functions on AAPL data."""
    import pandas as pd
    from pathlib import Path
    from features import prepare_all_features

    # Load and prepare data
    csv_path = Path('aapl_test_data.csv')
    if not csv_path.exists():
        print("CSV not found")
        return

    df = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
    df = prepare_all_features(df)

    # Simple split
    X_train, y_train, X_val, y_val, X_test, y_test = simple_split(df)

    print("Simple Split Info:")
    print_split_info("Train", X_train)
    print_split_info("Validation", X_val)
    print_split_info("Test", X_test)

    # Verify chronological
    if X_train.index.max() < X_val.index.min() and X_val.index.max() < X_test.index.min():
        print("Chronological order verified")
    else:
        print("Chronological order violated")

    # Walk-forward splits
    print("\nWalk-Forward Splits (first 3):")
    for i, (train_df, test_df) in enumerate(walk_forward_split(df)):
        if i >= 3:
            break
        validate_no_leakage(train_df, test_df)
        print(f"Iteration {i+1}:")
        print_split_info("Train", train_df)
        print_split_info("Test", test_df)


if __name__ == "__main__":
    test_splits()







