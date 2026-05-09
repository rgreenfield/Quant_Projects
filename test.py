import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from data_collector import download_stock_data, clean_data, add_returns, save_data, load_data, plot_price_and_returns
from features import prepare_all_features
from splitter import simple_split, walk_forward_split, validate_no_leakage, print_split_info

def main():
    end = datetime.today().date()
    start = end - timedelta(days=365 * 2)

    symbol = 'AAPL'
    filepath = 'aapl_test_data.csv'
    plot_path = 'aapl_test_plot.png'

    print('Downloading', symbol, 'from', start, 'to', end)
    df = download_stock_data(symbol, start_date=start.isoformat(), end_date=end.isoformat())
    print('Downloaded rows:', len(df))

    df = clean_data(df)
    df = add_returns(df)
    save_data(df, filepath)

    loaded = load_data(filepath)
    fig = plot_price_and_returns(loaded, symbol)
    fig.savefig(plot_path)

    df_features = prepare_all_features(loaded)
    print('Final DataFrame shape:', df_features.shape)
    print('Feature columns:', df_features.columns.tolist())
    print('Any NaN values?', df_features.isna().any().any())


def test_splits():
    """Test the splitting functions on AAPL data."""

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


if __name__ == '__main__':
    main()
    print("\n" + "="*50)
    test_splits()


