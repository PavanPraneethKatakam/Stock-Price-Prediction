import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime, timedelta
from textblob import TextBlob
import requests
from scipy import stats
from bs4 import BeautifulSoup

# ... [previous helper functions remain the same] ...

def generate_future_dates(end_date, prediction_days):
    """Generate future dates for prediction"""
    future_dates = pd.date_range(start=end_date, periods=prediction_days + 1)[1:]
    return future_dates

def prepare_future_features(df, prediction_days):
    """Prepare feature set for future prediction"""
    last_row = df.iloc[-1:]
    future_data = pd.DataFrame(index=range(prediction_days))
    
    # Copy the last known values for technical indicators
    for col in df.columns:
        future_data[col] = last_row[col].values[0]
    
    # Adjust features that should be different for each future day
    future_data['Price_Change'] = 0
    future_data['Volume_Change'] = 0
    future_data['Sentiment'] = df['Sentiment'].mean()  # Use average sentiment
    
    return future_data

def predict_future(model, df, features, prediction_days):
    """Generate future predictions"""
    future_data = prepare_future_features(df, prediction_days)
    
    if isinstance(model, Sequential):  # For LSTM model
        future_data_reshaped = np.reshape(future_data[features].values, 
                                        (future_data.shape[0], len(features), 1))
        predictions = model.predict(future_data_reshaped).flatten()
    else:  # For Random Forest model
        predictions = model.predict(future_data[features])
    
    return predictions

def main():
    st.title('Stock Price Prediction and Analysis')

    ticker = st.text_input('Enter Stock Ticker', 'AAPL')
    start_date = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
    end_date = st.date_input('End Date', datetime.now())
    prediction_days = st.slider('Number of days to predict into future', 1, 90, 30)

    if st.button('Analyze'):
        df = get_stock_data(ticker, start_date, end_date)
        df = add_technical_indicators(df)
        df = engineer_features(df, ticker)

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50', 
                   'RSI', 'MACD', 'Price_Change', 'Volume_Change', 'High_Low_Diff', 'Sentiment']
        X = df[features]
        y = df['Close']

        # Train models and get historical predictions
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LSTM': create_lstm_model((len(features), 1))
        }

        results = {}
        future_predictions = {}
        
        for name, model in models.items():
            # Train and evaluate on historical data
            model, mse, mape, r2, test_index, y_test, predictions = train_evaluate_model(X, y, model)
            results[name] = {'MSE': mse, 'MAPE': mape, 'R2': r2, 'Predictions': predictions}
            
            # Generate future predictions
            future_pred = predict_future(model, df, features, prediction_days)
            future_predictions[name] = future_pred

        # Generate future dates
        future_dates = generate_future_dates(df.index[-1], prediction_days)

        # Plot historical and future predictions
        st.subheader('Historical and Future Price Predictions')
        fig = go.Figure()
        
        # Plot historical prices
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], 
                               name='Historical Prices',
                               line=dict(color='blue')))

        # Plot future predictions for each model
        colors = ['red', 'green']
        for (name, predictions), color in zip(future_predictions.items(), colors):
            fig.add_trace(go.Scatter(x=future_dates, y=predictions,
                                   name=f'{name} Future Predictions',
                                   line=dict(color=color, dash='dash')))

        # Add confidence intervals (using standard deviation of predictions)
        all_predictions = np.array(list(future_predictions.values()))
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)

        fig.add_trace(go.Scatter(x=future_dates, y=mean_pred + 2*std_pred,
                               fill=None, mode='lines', line=dict(color='gray', width=0),
                               showlegend=False))
        fig.add_trace(go.Scatter(x=future_dates, y=mean_pred - 2*std_pred,
                               fill='tonexty', mode='lines', line=dict(color='gray', width=0),
                               name='95% Confidence Interval'))

        # Update layout
        fig.update_layout(title=f'{ticker} Stock Price Prediction',
                         xaxis_title='Date',
                         yaxis_title='Price',
                         hovermode='x unified')
        st.plotly_chart(fig)

        # Display prediction statistics
        st.subheader('Future Price Statistics')
        future_stats = pd.DataFrame({
            'Date': future_dates,
            'Mean Prediction': mean_pred,
            'Lower Bound': mean_pred - 2*std_pred,
            'Upper Bound': mean_pred + 2*std_pred
        })
        st.write(future_stats)

        # Risk disclaimer
        st.warning("""
        Disclaimer: Future predictions are based on historical patterns and available data. 
        They should not be used as the sole basis for investment decisions. Various external 
        factors and market conditions can significantly impact actual stock performance.
        """)

if __name__ == '__main__':
    main()
