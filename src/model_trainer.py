import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score  # Added to calculate accuracy
import joblib
import os

from src.feature_engineering import feature_engineering  # Import feature engineering

def build_and_save_model(data_path, model_output_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Apply feature engineering
    df = feature_engineering(df)

    # Features and target
    X = df[['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap',
            'volume_to_marketcap', 'volatility_1h', 'volatility_24h',
            'volatility_7d', 'price_ma3']]
    y = df['liquidity_ratio']  # Ensure this column exists and is spelled correctly

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Transform test set as well

    # XGBoost + GridSearch
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [300],
        'max_depth': [4, 6],
        'learning_rate': [0.1],
        'subsample': [0.9],
        'colsample_bytree': [0.9]
    }
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_xgb = grid_search.best_estimator_

    # Bagging
    bagging_model = BaggingRegressor(estimator=best_xgb, n_estimators=10, random_state=42, n_jobs=-1)
    bagging_model.fit(X_train_scaled, y_train)

    # Model prediction
    y_pred = bagging_model.predict(X_test_scaled)

    # Calculate R² score (accuracy)
    accuracy = r2_score(y_test, y_pred)

    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(bagging_model, model_output_path)
    print(f"Model saved to {model_output_path}")

    # Return accuracy (R² score)
    return round(accuracy, 4)  # Rounded for readability
