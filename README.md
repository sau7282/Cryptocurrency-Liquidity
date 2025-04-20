
# Cryptocurrency Liquidity Prediction

This project is a machine learning application built using **Flask** to predict the **liquidity ratio** of cryptocurrencies based on historical market data. It includes data preprocessing, feature engineering, model training using XGBoost + Bagging, and prediction interfaces.

## 🚀 Features

- Flask web interface for user input and predictions  
- Model training with XGBoost + BaggingRegressor and GridSearchCV  
- Data validation, preprocessing, and feature engineering pipeline  
- HTML-based result pages for both training and prediction  
- Logging with timestamps  
- JSON and HTML response support  

---

## 📁 Project Structure

Cryptocurrency-Liquidity/ │├── app.py # Main Flask application ├── requirements.txt # Python dependencies ├── README.md # Project documentation │ ├── data/ │ └── cleaned_cryptocurrency_data.csv # Input dataset │ ├── models/ │ └── trained_model.pkl # Trained ML model │ ├── logging/ │ └── log_YYYY-MM-DD_HH-MM-SS.log # Runtime logs │ ├── templates/ │ ├── index.html # Input form page │ ├── prediction_result.html # Output page for prediction │ └── train_result.html # Output page for training accuracy │ └── src/ ├── data_preprocessing.py # Input scaling and transformation ├── data_validation.py # Input validation ├── feature_engineering.py # Feature engineering logic ├── model_trainer.py # Model training + saving └── predict.py # Prediction using trained model

yaml
Copy
Edit

---

## 🧠 Model Info

- **Algorithm**: XGBoost Regressor wrapped with Bagging  
- **GridSearchCV**: Hyperparameter tuning (`max_depth`, `n_estimators`, etc.)  
- **Evaluation Metric**: R² Score  
- **Preprocessing**: StandardScaler for feature scaling  

---

## 🌐 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Cryptocurrency-Liquidity.git
cd Cryptocurrency-Liquidity
2. Create & activate a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the Flask app
bash
Copy
Edit
python app.py
Visit http://127.0.0.1:5000/ in your browser.

💡 Usage
Prediction
Fill in the crypto stats on the homepage.

Click "Get Prediction".

View the predicted liquidity ratio and click "Go Back" to enter new values.

Model Training
Use a POST request or route trigger /train.

View training accuracy on the train_result.html page.

🛠 Dependencies
Flask

pandas

scikit-learn

xgboost

joblib

Make sure these are listed in your requirements.txt.

📌 Notes
Ensure your input CSV is correctly formatted and includes all required columns.

You can update model_trainer.py to add additional evaluation metrics if needed.

📜 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Saurabh Kumar
