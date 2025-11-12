import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from flask import Flask, jsonify
from flask_cors import CORS

PREDICTION_DAYS = 5 

app = Flask(__name__)
CORS(app) 

def fetch_data(ticker):
    try:
        data = yf.download(ticker, period="5y")
        if data.empty:
            raise ValueError("No data found.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def preprocess_data(data):
    df = data[['Close']].copy()

    for i in range(1, PREDICTION_DAYS + 1):
        df[f'Lag_{i}'] = df['Close'].shift(i)

    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df.drop(['Target', 'Close'], axis=1)
    y = df['Target']
    last_features = df.drop('Target', axis=1).iloc[-1]
    last_close_price = data['Close'].iloc[-1].item()

    X = X[:-1]
    y = y[:-1]
    
    return X, y, last_features, last_close_price

def train_and_predict(X, y, last_features):
    print("--- Training Model...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)

    last_features_array = last_features.drop('Close').values.reshape(1, -1)
    final_prediction = model.predict(last_features_array)[0]

    return final_prediction

@app.route('/predict/<ticker>', methods=['GET'])
def predict_stock(ticker):
    ticker = ticker.upper().strip()

    data = fetch_data(ticker)
    if data is None or data.empty:
        return jsonify({
            "error": "Data Not Found", 
            "message": f"Could not fetch historical data for ticker: {ticker}."
        }), 404

    try:
        X, y, last_features, last_close_price = preprocess_data(data)
    except Exception as e:
         return jsonify({
            "error": "Preprocessing Error", 
            "message": "Data preprocessing failed. Check data quality."
        }), 500

    final_prediction = train_and_predict(X, y, last_features)
    
    chart_series = data['Close'].tail(50) 
    
    response_data = {
        "ticker": ticker,
        "last_close_price": float(last_close_price),
        "predicted_close_price": float(final_prediction),
        "last_date": data.index[-1].strftime('%m/%d/%Y'),
        
        "historical_prices": chart_series.tolist(), 
        "historical_dates": [date.strftime('%m/%d') for date in chart_series.index]
    }

    return jsonify(response_data)

if __name__ == '__main__':
    print("Starting Flask API...")
    app.run(debug=True, port=5000)