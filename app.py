import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Project modules
from src.feature_engineering import feature_engineering
from src.model_trainer import build_and_save_model

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Initialize Flask app
app = Flask(__name__)

# Set up logging
log_dir = "logging"
os.makedirs(log_dir, exist_ok=True)

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
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['features']
    logging.info("Bagging model loaded successfully.")
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

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Feature Engineering (set is_training=False for prediction)
        df = feature_engineering(df, is_training=False)

        # Ensure only the required features are used
        df = df[feature_columns]

        # Scale the input data
        df_scaled = scaler.transform(df)

        # Predictions
        bagging_prediction = model.predict(df_scaled)[0]

        app.logger.info(f"Bagging Prediction: {bagging_prediction}")

        # Render the HTML page with the prediction results
        return render_template(
            'prediction_result.html',
            bagging_prediction=round(bagging_prediction, 4)
        )

    except Exception as e:
        app.logger.error(f"Error occurred during prediction: {e}")
        return render_template('error.html', message=str(e)), 500

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
    
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('error.html', message="No file uploaded for batch prediction.")

        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', message="No file selected for batch prediction.")

        # Determine file type and read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return render_template('error.html', message="Unsupported file format. Please upload a CSV or Excel file.")

        app.logger.debug(f"Batch prediction data: {df.head()}")

        # Apply feature engineering
        df = feature_engineering(df, is_training=False)

        # Ensure only the required features are used
        df = df[feature_columns]

        # Scale the input data
        df_scaled = scaler.transform(df)

        # Batch predictions
        predictions = model.predict(df_scaled)
        df['predictions'] = predictions

        # Save the results to a CSV file
        output_csv = df.to_csv(index=False)
        return Response(
            output_csv,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=batch_predictions.csv"}
        )

    except Exception as e:
        app.logger.error(f"Error occurred during batch prediction: {e}")
        return render_template('error.html', message=str(e)), 500

if __name__ == "__main__":
    app.logger.info("Starting Flask app...")
    app.run(debug=True)