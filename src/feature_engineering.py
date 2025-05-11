def feature_engineering(data, is_training=True):
    """
    Adds features for real-time single-row prediction.
    Assumes `data` is a pandas DataFrame.
    """
    # Ratio of volume to market cap
    data['volume_to_marketcap'] = data['24h_volume'] / data['mkt_cap']
    
    # Volatility features
    data['volatility_1h'] = data['price'] * (data['1h'] / 100)
    data['volatility_24h'] = data['price'] * (data['24h'] / 100)
    data['volatility_7d'] = data['price'] * (data['7d'] / 100)

    # Moving average (3-day)
    data['price_ma3'] = data['price'].rolling(window=3, min_periods=1).mean()

    # Liquidity ratio (only for training)
    if is_training:
        data['liquidity_ratio'] = data['24h_volume'] / data['mkt_cap']

    return data