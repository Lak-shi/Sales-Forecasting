# Supply Chain Demand Forecasting

## 📌 Project Overview
This project implements a **hybrid demand forecasting** solution using **LSTM, Prophet, and XGBoost** to predict sales for a supply chain dataset. The goal is to compare different forecasting techniques and blend them to achieve optimal accuracy.

## 🚀 Features
- **Data Extraction & Processing**
  - Fetch data from BigQuery
  - Preprocess data (feature engineering, handling missing values, etc.)
- **Forecasting Models**
  - **LSTM (Long Short-Term Memory)**: Captures sequential patterns in time-series data.
  - **Prophet**: Captures long-term trends and seasonality.
  - **XGBoost**: Provides high-accuracy gradient boosting-based predictions.
  - **Hybrid Model**: Blends Prophet’s trend prediction with XGBoost’s fine-grained pattern recognition.
- **Hyperparameter Tuning**
  - GridSearchCV for LSTM tuning
  - Optuna for XGBoost hyperparameter optimization
- **Model Evaluation**
  - RMSE, MAPE, R² Score comparison
- **Deployment**
  - Upload forecasts to BigQuery
  - Automated Airflow DAGs

## 📊 Model Performance Comparison
| Model    | RMSE  | R² Score |
|----------|-------|----------|
| **LSTM**  | 0.1333 | 0.2168   |
| **Prophet** | 0.1032 | 0.5328   |
| **Hybrid (LSTM + Prophet)** | 0.1433 | 0.0991   |
| **XGBoost** | 6.5965 | 0.9476   |
| **Hybrid (Prophet + XGBoost)** | 6.9682 | 0.9413   |

## 📂 Project Structure
```
supplychain_forecasting/
│── airflow/
│   ├── dags/
│   │   ├── bigquery_key.json
│   │   ├── postgres_to_bigquery.py
│── data/
│   ├── sales_data.csv
│   ├── sample_submission.csv
│   ├── test.csv
│── models/
│   ├── hybrid_xgboost_prophet.model
│   ├── lstm_sales_forecast.h5
│   ├── optimized_xgboost_optuna.model
│── scripts/
│   ├── fetch_bigquery.py
│   ├── preprocess_data.py
│   ├── train_lstm.py
│   ├── train_xgboost.py
│   ├── train_prophet_extract_trend.py
│   ├── evaluate_models.py
│   ├── upload_to_bigquery.py
│── README.md
│── requirements.txt
│── .gitignore
```

## 📜 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/supplychain_forecasting.git
   cd supplychain_forecasting
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 🔄 Usage
1. **Preprocess Data**
   ```sh
   python scripts/preprocess_data.py
   ```
2. **Train Models**
   ```sh
   python scripts/train_lstm.py
   python scripts/train_xgboost.py
   python scripts/train_prophet_extract_trend.py
   ```
3. **Evaluate Models**
   ```sh
   python scripts/evaluate_models.py
   ```
4. **Deploy Forecasts**
   ```sh
   python scripts/upload_to_bigquery.py
   ```

## 📌 Dependencies
- Python 3.11
- TensorFlow / Keras
- XGBoost
- Prophet
- Pandas, NumPy, Scikit-learn
- Google Cloud BigQuery SDK
- Airflow

## 📬 Contact
For any questions, feel free to reach out!
- **Email**: your.email@example.com
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourusername)

---
🛠️ Built with **Machine Learning & AI** to enhance supply chain forecasting!

