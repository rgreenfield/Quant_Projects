
from datetime import datetime, timedelta
from data_collector import download_stock_data, clean_data, add_returns, save_data, load_data, plot_price_and_returns

end = datetime.today().date()
start = end - timedelta(days=365*2)

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

"""
print('Saved CSV to', filepath)
print('Saved plot to', plot_path)
print('Loaded shape:', loaded.shape)
print('Missing values by column:')
print(loaded.isna().sum())
print('First 5 rows:')
print(loaded.head(5))
print('Any missing values?', loaded.isna().any().any())

"""


