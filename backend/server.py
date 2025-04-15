from datetime import timedelta
from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import os
from scipy.stats import norm
from arch import arch_model

app = Flask(__name__)
CORS(app)

# Define a global flag to track if models are loaded
models_loaded = False
lstm_model = None
cnn_model = None

def create_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(100, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(100, 1)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Try to load models, or create new ones if loading fails
def initialize_models():
    global lstm_model, cnn_model, models_loaded
    
    try:
        # First attempt - try with custom objects
        class LSTMWithTimeMajor(tf.keras.layers.LSTM):
            def __init__(self, *args, time_major=False, **kwargs):
                super().__init__(*args, **kwargs)
                
        custom_objects = {
            'LSTMWithTimeMajor': LSTMWithTimeMajor,
            # Add other custom layers if needed
        }
        
        lstm_model_path = './models/lstm_model_adam.h5'
        cnn_model_path = './models/diluted_cnn.h5'
        
        if os.path.exists(lstm_model_path):
            try:
                print("Attempting to load LSTM model with custom objects...")
                lstm_model = tf.keras.models.load_model(lstm_model_path, custom_objects=custom_objects)
                print("LSTM model loaded successfully!")
            except Exception as e:
                print(f"Could not load LSTM model with custom objects: {e}")
                print("Creating new LSTM model...")
                lstm_model = create_lstm_model()
                # Optionally save the new model
                # lstm_model.save('./models/lstm_model_new.h5')
        else:
            print("LSTM model file not found, creating new model...")
            lstm_model = create_lstm_model()
            
        if os.path.exists(cnn_model_path):
            try:
                print("Attempting to load CNN model with custom objects...")
                cnn_model = tf.keras.models.load_model(cnn_model_path, custom_objects=custom_objects)
                print("CNN model loaded successfully!")
            except Exception as e:
                print(f"Could not load CNN model with custom objects: {e}")
                print("Creating new CNN model...")
                cnn_model = create_cnn_model()
                # Optionally save the new model
                # cnn_model.save('./models/cnn_model_new.h5')
        else:
            print("CNN model file not found, creating new model...")
            cnn_model = create_cnn_model()
        
        models_loaded = True
        print("Models initialization complete")
        
    except Exception as e:
        print(f"Error during model initialization: {e}")
        # Create new models as fallback
        print("Creating new models as fallback...")
        lstm_model = create_lstm_model()
        cnn_model = create_cnn_model()
        models_loaded = True

# Initialize models when server starts
print("Initializing models...")
initialize_models()

def load_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']]
    return df

def prepare_data(data, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

@app.route('/predict-lstm', methods=['POST'])
def predict():
    if not models_loaded or lstm_model is None:
        return jsonify({"error": "Model not available. Server initialization failed."})
        
    content = request.json
    ticker = content['ticker']
    start_date = content['start_date']
    end_date = content['end_date']
    look_back = 100  # Fixed look_back period

    df = load_data(ticker, start_date, end_date)
    if df.isnull().any().any():
        df = df.dropna()  # Drop any rows with NaN values

    if df.empty:
        return jsonify({"error": "No data available for the given ticker and date range."})
    future_days = 10
    X, y, scaler = prepare_data(df['Close'], look_back=100)
    
    try:
        predictions = lstm_model.predict(X)
        predictions_rescaled = scaler.inverse_transform(predictions)
        y_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

        dates = df.index[look_back:].tolist()  # List of dates corresponding to the predictions
        dates = [date.strftime('%Y-%m-%d') for date in dates] 

        mse = mean_squared_error(y_rescaled, predictions_rescaled)
        mae = mean_absolute_error(y_rescaled, predictions_rescaled)
        r2 = r2_score(y_rescaled, predictions_rescaled)
        mape = mean_absolute_percentage_error(y_rescaled, predictions_rescaled)
        accuracy = 100 - mape
        last_batch = X[-1:].reshape(1, 100, -1)
        future_predictions = []

        stock_list = []
        x1 = y_rescaled.flatten().tolist()
        x2 = predictions_rescaled.flatten().tolist()
        for i in range(len(x2)):
            stock_list.append([dates[i], x1[i], x2[i]])

        for _ in range(future_days):
            current_pred = lstm_model.predict(last_batch)
            current_pred_rescaled = scaler.inverse_transform(current_pred).flatten()[0]
            future_predictions.append(float(current_pred_rescaled))  # Convert NumPy float to Python float
            last_batch = np.append(last_batch[:, 1:, :], current_pred.reshape(1, 1, -1), axis=1)

        final_dates = [pd.to_datetime(end_date) + timedelta(days=i+1) for i in range(future_days)]
        final_dates_str = [date.strftime('%Y-%m-%d') for date in final_dates]
        stock_l = [[date, None, pred] for date, pred in zip(final_dates_str, future_predictions)]
        stock_list.extend(stock_l)
        response = {
            "ticker": ticker,
            "stock_data": stock_list,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "accuracy": accuracy
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"})


@app.route('/predict-cnn', methods=['POST'])
def predictCnn():
    if not models_loaded or cnn_model is None:
        return jsonify({"error": "Model not available. Server initialization failed."})

    content = request.json
    ticker = content['ticker']
    start_date = content['start_date']
    end_date = content['end_date']
    look_back = 100  # Fixed look_back period

    df = load_data(ticker, start_date, end_date)
    if df.isnull().any().any():
        df = df.dropna()  # Drop any rows with NaN values

    if df.empty:
        return jsonify({"error": "No data available for the given ticker and date range."})
    
    future_days = 10
    X, y, scaler = prepare_data(df['Close'], look_back=100)
    
    try:
        predictions = cnn_model.predict(X)
        predictions_rescaled = scaler.inverse_transform(predictions)
        y_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

        dates = df.index[look_back:].tolist()  # List of dates corresponding to the predictions
        dates = [date.strftime('%Y-%m-%d') for date in dates] 

        mse = mean_squared_error(y_rescaled, predictions_rescaled)
        mae = mean_absolute_error(y_rescaled, predictions_rescaled)
        r2 = r2_score(y_rescaled, predictions_rescaled)
        mape = mean_absolute_percentage_error(y_rescaled, predictions_rescaled)
        accuracy = 100 - mape
        last_batch = X[-1:].reshape(1, 100, -1)
        future_predictions = []

        stock_list = []
        x1 = y_rescaled.flatten().tolist()
        x2 = predictions_rescaled.flatten().tolist()
        for i in range(len(x2)):
            stock_list.append([dates[i], x1[i], x2[i]])

        for _ in range(future_days):
            # Note: Using CNN model here instead of lstm_model for predictions
            current_pred = cnn_model.predict(last_batch)
            current_pred_rescaled = scaler.inverse_transform(current_pred).flatten()[0]
            future_predictions.append(float(current_pred_rescaled))  # Convert NumPy float to Python float
            last_batch = np.append(last_batch[:, 1:, :], current_pred.reshape(1, 1, -1), axis=1)

        final_dates = [pd.to_datetime(end_date) + timedelta(days=i+1) for i in range(future_days)]
        final_dates_str = [date.strftime('%Y-%m-%d') for date in final_dates]
        stock_l = [[date, None, pred] for date, pred in zip(final_dates_str, future_predictions)]
        stock_list.extend(stock_l)
        response = {
            "ticker": ticker,
            "stock_data": stock_list,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "accuracy": accuracy
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"})


@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    request_data = request.get_json()
    ticker = request_data.get('ticker')
    period = request_data.get('period')
    if not ticker or not period:
        return jsonify({"error": "Please provide both 'ticker' and 'period'"}), 400
    data = yf.Ticker(ticker)
    hist = data.history(period=period)
    hist['DateTime'] = hist.index.strftime('%Y-%m-%d')
    if hist.empty:
        return jsonify({"error": "No data found"}), 404
    columns_order = ['DateTime', 'Low', 'Open', 'Close', 'High']
    ordered_data = hist.reset_index()[columns_order].values.tolist()
    return jsonify({"data": ordered_data})

@app.route('/sip-calculator', methods=['POST'])
def sip_calculator():
    request_data = request.get_json()
    sip_amount = float(request_data.get('sip_amount'))
    sip_period = int(request_data.get('sip_period'))
    sip_rate = float(request_data.get('sip_rate'))
    monthly_rate = sip_rate / 12 / 100
    months = sip_period * 12
    future_value = 0
    for _ in range(months):
        future_value = (future_value + sip_amount) * (1 + monthly_rate)
    invested_amount = sip_amount * months
    return jsonify({"returns": [invested_amount, future_value-invested_amount]})


def fetch_stock_data(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return stock_data

def train_linear_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, train_size=0.8)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


@app.route('/predict-linear', methods=['POST'])
def predict_and_plot_stock():
    request_data = request.get_json()
    ticker_symbol = request_data['ticker']
    start_date = request_data['start_date']
    end_date = request_data['end_date']

    try:
        stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)
        if stock_data.empty:
            return jsonify({"error": "No data available for the given ticker and date range."})
            
        dates = stock_data.index.tolist()  # List of dates corresponding to the predictions
        X = np.arange(len(stock_data)).reshape(-1, 1)  # Using index as a feature
        y = stock_data['Close'].values  # Target variable

        model, X_test, y_test = train_linear_regression_model(X, y)
        predictions = model.predict(X)  # Predict for all data points
        
        # Calculate metrics using the test portion
        test_indices = list(range(len(X)))[len(X_train):]
        test_predictions = predictions[test_indices]
        mse = mean_squared_error(y_test, test_predictions)
        
        dates_str = [date.strftime('%Y-%m-%d') for date in dates]
        stock_list = [[dates_str[i], float(y[i]), float(predictions[i])] for i in range(len(X))]
        
        return jsonify({
            'ticker': ticker_symbol,
            'mean_squared_error': float(mse),
            "stock_data": stock_list
        })
    except Exception as e:
        return jsonify({"error": f"Error during linear prediction: {str(e)}"})

@app.route('/predict-garch', methods=['POST'])
def predict_garch():
    request_data = request.get_json()
    ticker_symbol = request_data['ticker']
    start_date = request_data['start_date']
    end_date = request_data['end_date']
    
    try:
        stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)
        if stock_data.empty:
            return jsonify({"error": "No data available for the given ticker and date range."})
            
        returns = 100 * stock_data['Close'].pct_change().dropna()
        model = arch_model(returns, vol='Garch', p=1, q=1)
        model_fit = model.fit(disp='off')
        forecast = model_fit.forecast(horizon=5)
        
        return jsonify({
            'ticker': ticker_symbol,
            'forecast': forecast.mean.iloc[-1].values.tolist()
        })
    except Exception as e:
        return jsonify({"error": f"Error during GARCH prediction: {str(e)}"})

def get_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            return None, None
        return hist['Close'].iloc[-1], hist['Close'].pct_change().dropna()
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None, None

def calculate_annualized_volatility(daily_returns):
    if daily_returns is None or len(daily_returns) == 0:
        return 0.2  # Default volatility if data is not available
    daily_volatility = np.std(daily_returns)
    annualized_volatility = daily_volatility * np.sqrt(252)
    return annualized_volatility

def black_scholes_call(S, K, T, r, sigma):
    try:
        S = float(S)
        K = float(K)
        T = float(T)
        r = float(r)
        sigma = float(sigma)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return call_price
    except Exception as e:
        print(f"Error in Black-Scholes calculation: {e}")
        return None


@app.route('/black-scholes', methods=['POST'])
def finance():
    try:
        data = request.json
        ticker = data['ticker']
        K = data['strike_price']
        r = data['risk_free_rate']
        T = data['time_to_maturity']
        sigma = data['volatility']

        # Fetching stock price and calculating historical volatility
        S, daily_returns = get_data(ticker)
        
        if S is None:
            return jsonify({"error": f"Could not fetch stock data for {ticker}"}), 400
            
        if sigma == 'auto':  # Use historical volatility if 'auto' is specified
            sigma = calculate_annualized_volatility(daily_returns)

        # Calculate Black-Scholes call option price
        call_option_price = black_scholes_call(S, K, T, r, sigma)
        
        if call_option_price is None:
            return jsonify({"error": "Error calculating option price"}), 400
            
        return jsonify({
            "ticker": ticker,
            "current_price": float(S),
            "call_option_price": float(call_option_price),
            "annualized_volatility": float(sigma) * 100  # Convert to percentage
        })
    except Exception as e:
        return jsonify({"error": f"Error in Black-Scholes calculation: {str(e)}"}), 400


if __name__ == '__main__':
    app.run(debug=True)