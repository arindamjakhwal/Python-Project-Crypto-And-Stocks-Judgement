import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np
import time

# --- Deep Learning Libraries ---
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout 

# --- TensorFlow Environment Fix (Addressing previous freezing issue) ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.experimental.set_visible_devices([], 'GPU')
# ----------------------------------------------------------------------

# --- Configuration and Initialization ---
st.set_page_config(
    page_title="Algorithmic Financial Predictor (LSTM)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_TICKERS = "BTC-USD, AAPL, TSLA"
TIME_STEP = 60 # Number of historical days the model looks back to make a single prediction
TRAINING_EPOCHS = 20 # Number of training iterations (keep low for fast execution in app)

# Function to safely fetch data with caching
@st.cache_data(ttl=600)
def get_stock_data(tickers, start_date, end_date):
    """Fetches historical OHLC data for a list of tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(',')]
    
    try:
        data = yf.download(
            tickers=ticker_list,
            start=start_date,
            end=end_date,
            interval="1d",
            group_by='ticker' if len(ticker_list) > 1 else 'column',
            threads=True
        )
        if len(ticker_list) == 1:
            data.columns = pd.MultiIndex.from_product([[ticker_list[0]], data.columns])
        
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Error fetching data for tickers: {e}")
        return pd.DataFrame()

# Helper function to prepare data for LSTM (create sequences)
def create_dataset(data, time_step=TIME_STEP):
    """Converts time-series data into sequences of [X (past 60 days), Y (next day)]"""
    X_data, Y_data = [], []
    for i in range(len(data) - time_step - 1):
        # Sequence of 60 days
        a = data[i:(i + time_step), 0]
        X_data.append(a)
        # Next day's price (the target)
        Y_data.append(data[i + time_step, 0])
    return np.array(X_data), np.array(Y_data)

# --- Full Prediction Module (The Intense Part) ---
def run_prediction_algorithm(data_series, prediction_days):
    """
    Implements data scaling, LSTM model building/training, and multi-day forecasting.
    """
    if len(data_series) < TIME_STEP + 1:
        st.warning(f"Not enough historical data ({len(data_series)} days). Need at least {TIME_STEP + 1} days to train the model.")
        return pd.DataFrame(), None

    # Use only the 'Close' price for the time series
    raw_data = data_series['Close'].values.reshape(-1, 1)
    
    # 1. Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(raw_data)
    
    # 2. Create Sequences (X, Y)
    X_train, Y_train = create_dataset(scaled_data, TIME_STEP)
    
    # Reshape input to be [samples, time steps, features] for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # 3. Build the LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(TIME_STEP, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Output layer predicts one day ahead
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 4. Train the Model (Wrapped in error handling)
    try:
        model.fit(X_train, Y_train, epochs=TRAINING_EPOCHS, batch_size=32, verbose=0)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return pd.DataFrame(), None
    
    # 5. Multi-Day Forecasting
    
    # Get the last 60 days of training data to use as the first prediction input
    temp_input = scaled_data[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    
    predicted_prices_scaled = []
    
    # Loop to predict 'prediction_days' ahead
    for _ in range(prediction_days):
        # Make prediction
        next_prediction = model.predict(temp_input, verbose=0)[0]
        
        # Store prediction
        predicted_prices_scaled.append(next_prediction)
        
        # Update input array: shift window by one, append new prediction
        # Get the current window (which is 60 steps long)
        current_window = temp_input[0] 
        # Remove the oldest point and append the newest prediction
        new_window = np.append(current_window[1:], next_prediction) 
        # Update the input array for the next day's prediction
        temp_input = new_window.reshape(1, TIME_STEP, 1)

    # 6. Inverse Transform and Prepare Output
    
    # Inverse transform to get actual dollar values
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1)).flatten()
    
    # Generate future dates
    last_date = data_series.index[-1].date()
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
    
    # Prepare DataFrame
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': predicted_prices})
    forecast_df = forecast_df.set_index('Date')
    
    # Predict the last training price for smooth chart transition
    predicted_last_train = scaler.inverse_transform(model.predict(X_train[-1].reshape(1, TIME_STEP, 1), verbose=0)).flatten()[0]

    return forecast_df, predicted_last_train

# --- Streamlit UI Layout ---

st.title("📈 Deep Learning Financial Price Predictor (LSTM)")
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("Data Configuration")
ticker_input = st.sidebar.text_input(
    "Enter Ticker Symbols (e.g., AAPL, BTC-USD)", 
    DEFAULT_TICKERS
)

# Date range selector for historical data
today = date.today()
default_start = today - timedelta(days=730) # 2 years of data for better training
date_start = st.sidebar.date_input("Start Date", default_start)
date_end = st.sidebar.date_input("End Date", today)

# Prediction control
prediction_days = st.sidebar.slider(
    "Prediction Horizon (Days Ahead)",
    min_value=1,
    max_value=30,
    value=7
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"Model Configuration: LSTM looks at the past **{TIME_STEP} days** of data and trains for **{TRAINING_EPOCHS} epochs**."
)

# --- Main Content Logic ---
if ticker_input:
    # 1. Fetch Data
    with st.spinner('Fetching historical data...'):
        historical_data = get_stock_data(ticker_input, date_start, date_end)
    
    if historical_data.empty:
        st.error("Could not load data for the specified tickers. Please check the symbols or date range.")
    else:
        tickers_to_process = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

        # Allow user to select which asset to analyze
        if len(tickers_to_process) > 1:
            selected_ticker = st.selectbox(
                "Select Asset for Detailed View and Prediction", 
                options=tickers_to_process
            )
        else:
            selected_ticker = tickers_to_process[0]
            st.header(f"Analysis for {selected_ticker}")

        # Extract the data for the selected ticker
        if selected_ticker in historical_data.columns.get_level_values(0):
            ticker_data = historical_data[selected_ticker].copy()
            
            # --- Tab 1: Live Data & Summary ---
            st.subheader("Current Market Summary")
            
            latest_data = ticker_data.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Current Close Price", f"${latest_data['Close']:.2f}")
            col2.metric("Previous Day Open", f"${latest_data['Open']:.2f}")
            col3.metric("Daily High", f"${latest_data['High']:.2f}")
            col4.metric("Trading Volume", f"{latest_data['Volume']:,}")
            
            st.markdown("---")
            
            # --- Tab 2: Prediction Module Execution ---
            st.header("🔮 LSTM Prediction Results")

            with st.spinner(f"Training LSTM model ({TRAINING_EPOCHS} epochs) and forecasting {prediction_days} days..."):
                start_time = time.time()
                forecast_df, predicted_last_train = run_prediction_algorithm(ticker_data, prediction_days)
                end_time = time.time()
            
            if not forecast_df.empty:
                st.success(f"Prediction complete in {end_time - start_time:.2f} seconds.")
                st.info(f"The model is predicting prices for the next **{prediction_days} days**.")

                # ------------------ Visualization ------------------
                
                # Create a combined DataFrame for plotting
                plot_data = ticker_data[['Close']].copy()
                plot_data = plot_data.rename(columns={'Close': 'Actual Price'})
                
                # Add a marker for the predicted price of the last training day
                predicted_actual_df = pd.DataFrame({
                    'Actual Price': [predicted_last_train],
                    'Predicted Price': [predicted_last_train]
                }, index=[plot_data.index[-1]])
                
                # Combine historical actuals, the predicted forecast, and the last predicted training point
                combined_df = pd.concat([plot_data, predicted_actual_df], axis=0).sort_index()

                # Add the forecast data
                combined_df = pd.concat([combined_df, forecast_df], axis=0).sort_index()
                
                # Use Plotly Graph Objects for more control over traces and gaps
                fig_pred = go.Figure()

                # 1. Actual Price Trace
                fig_pred.add_trace(go.Scatter(
                    x=plot_data.index, y=plot_data['Actual Price'],
                    mode='lines', name='Historical Actual',
                    line=dict(color='blue')
                ))

                # 2. Forecast Trace
                last_hist_date = plot_data.index[-1]
                
                # Define x and y for the continuous forecast line (starting from the last actual point)
                forecast_x = [last_hist_date] + forecast_df.index.tolist()
                forecast_y = [plot_data['Actual Price'].iloc[-1]] + forecast_df['Predicted Price'].tolist()
                
                fig_pred.add_trace(go.Scatter(
                    x=forecast_x, y=forecast_y,
                    mode='lines+markers', name='LSTM Forecast',
                    line=dict(color='orange', dash='dash'),
                    marker=dict(size=5)
                ))
                
                # Layout updates
                fig_pred.update_layout(
                    title=f'Historical Price vs. {prediction_days}-Day LSTM Forecast for {selected_ticker}',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=600,
                    hovermode="x unified"
                )
                
                fig_pred.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                st.subheader("Predicted Prices Table")
                st.dataframe(forecast_df)
                
                # --- Historical Chart (optional, can be simplified or removed) ---
                st.subheader(f"Historical Price Chart ({selected_ticker})")
                fig_hist = px.line(
                    ticker_data, 
                    y='Close', 
                    title=f'Close Price History for {selected_ticker}',
                    labels={'Close': 'Price (USD)', 'index': 'Date'},
                    height=300
                )
                fig_hist.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.warning("Please enter at least one ticker symbol in the sidebar to begin.")