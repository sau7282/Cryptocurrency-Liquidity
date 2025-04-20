def validate_input_data(data):
    # List of required columns (features)
    required_columns = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap']

    # Check if all required columns are present in the data
    if not all(col in data for col in required_columns):
        return False, "Missing one or more required fields."

    # Validate if each column contains a valid numeric value
    try:
        price = float(data['price'])
        one_hour = float(data['1h'])
        twenty_four_hour = float(data['24h'])
        seven_day = float(data['7d'])
        volume = float(data['24h_volume'])
        market_cap = float(data['mkt_cap'])
    except ValueError:
        return False, "All fields must contain valid numeric values."

    # Additional checks for positive values (if required)
    if price <= 0 or one_hour <= 0 or twenty_four_hour <= 0 or seven_day <= 0 or volume <= 0 or market_cap <= 0:
        return False, "All fields must have positive numeric values."

    # All checks passed
    return True, ""
