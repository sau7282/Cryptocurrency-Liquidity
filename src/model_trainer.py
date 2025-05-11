import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import os
from src.feature_engineering import feature_engineering

def build_and_save_model(data_path, model_output_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Apply feature engineering
    df = feature_engineering(df)

    # Features and target
    feature_columns = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap',
                       'volume_to_marketcap', 'volatility_1h', 'volatility_24h',
                       'volatility_7d', 'price_ma3']
    X = df[feature_columns]
    y = df['liquidity_ratio']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled data back to DataFrame to retain feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)

    # XGBoost + Bagging
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    bagging_model = BaggingRegressor(estimator=xgb, n_estimators=10, random_state=42, n_jobs=-1)
    bagging_model.fit(X_train_scaled, y_train)

    # Model prediction
    y_pred = bagging_model.predict(X_test_scaled)

    # Calculate R² score
    accuracy = r2_score(y_test, y_pred)

    # Save model and scaler
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump({'model': bagging_model, 'scaler': scaler, 'features': feature_columns}, model_output_path)
    print(f"Model saved to {model_output_path}")

    return round(accuracy, 4)