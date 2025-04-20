def feature_engineering(data):
    """
    Adds features for real-time single-row prediction.
    Assumes `data` is a pandas DataFrame with one row.
    """
    # Ratio of volume to market cap
    data['volume_to_marketcap'] = data['24h_volume'] / data['mkt_cap']
    
    # Volatility features
    data['volatility_1h'] = data['price'] * (data['1h'] / 100)
    data['volatility_24h'] = data['price'] * (data['24h'] / 100)
    data['volatility_7d'] = data['price'] * (data['7d'] / 100)

    # Fill moving average with current price if only 1 row
    data['price_ma3'] = data['price']

    return data
