import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

# Project modules
from src.predict import make_predictions
from src.data_preprocessing import preprocess_data
from src.data_validation import validate_input_data
from src.feature_engineering import feature_engineering
from src.model_trainer import build_and_save_model

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Initialize Flask app
app = Flask(__name__)

# Set up logging
log_dir = "logging"
os.makedirs(log_dir, exist_ok=True)
# log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")
# log_file = os.path.join(log_dir, log_filename)
log_file = 'logging/app.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)

# Paths
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'trained_model.pkl')
DATA_PATH = os.path.join(os.getcwd(), 'data', 'cleaned_cryptocurrency_data.csv')  

# Load or train model
try:
    if not os.path.exists(MODEL_PATH):
        logging.info("Model not found. Training a new model...")
        build_and_save_model(DATA_PATH, MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load or build model: {e}")
    sys.exit(1)

# Routes
@app.route('/')
def home():
    app.logger.info("Home route accessed.")
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = {
            "price": float(request.form['price']),
            "1h": float(request.form['1h']),
            "24h": float(request.form['24h']),
            "7d": float(request.form['7d']),
            "24h_volume": float(request.form['24h_volume']),
            "mkt_cap": float(request.form['mkt_cap']),
        }
        app.logger.debug(f"Received data: {data}")

        # Validate input
        if not validate_input_data(data):
            app.logger.warning("Invalid data format received.")
            return jsonify({"error": "Invalid input. Please check required features."}), 400
        
        # Preprocess
        preprocessed_data = preprocess_data(data)
        app.logger.debug(f"Data after preprocessing: {preprocessed_data}")

        # Feature Engineering
        features = feature_engineering(preprocessed_data)
        app.logger.debug(f"Features after engineering: {features}")

        # Prediction
        prediction = make_predictions(features, model)
        app.logger.info(f"Prediction made: {prediction}")

        # Render the HTML page with the prediction result
        return render_template('prediction_result.html', prediction=prediction[0])

    except Exception as e:
        app.logger.error(f"Error occurred during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Optional: Retrain via API
@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Train the model and return metrics
        accuracy = build_and_save_model(DATA_PATH, MODEL_PATH)

        # Render the HTML page with accuracy
        return render_template('train_result.html', accuracy=accuracy)

    except Exception as e:
        app.logger.error(f"Error occurred during model training: {e}")
        return render_template('error.html', message=str(e)), 500

if __name__ == "__main__":
    app.logger.info("Starting Flask app...")
    app.run(debug=True)
