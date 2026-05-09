from datetime import datetime, timedelta
from data_collector import download_stock_data, clean_data, add_returns, save_data, load_data, plot_price_and_returns
from features import prepare_all_features


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


if __name__ == '__main__':
    main()


