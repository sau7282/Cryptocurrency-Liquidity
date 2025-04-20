import pandas as pd

def preprocess_data(data):
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame([data])
    
    # Ensure the data types are correct
    df = df.astype({
        'price': float,
        '1h': float,
        '24h': float,
        '7d': float,
        '24h_volume': float,
        'mkt_cap': float
    })
    
    return df
