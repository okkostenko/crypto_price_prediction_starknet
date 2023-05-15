from data_collection.functions.data import get_data

btc = get_data("BTCUSDT", "1d", save=False)
print(len(btc))