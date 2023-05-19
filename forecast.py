import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

# Function to preprocess the data
def preprocess_data(df):
    # Convert to hourly interval
    df_hourly = df.resample('1H').mean().interpolate()

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_hourly)

    # Prepare the data for ARIMA model
    # User input for forecasting steps
    lookback = 2160
    X = []
    y = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler

# Function to build and train the ARIMA model
def build_model(X, y):
    # Flatten the input data
    X = X.reshape(X.shape[0], X.shape[1])
    
    model = ARIMA(y, order=(5, 1, 0))
    model_fit = model.fit()

    return model_fit

# Function to forecast data
def forecast_data(model, last_x, scaler, steps):
    forecast = model.forecast(steps)
    future_data = np.array(forecast[0])
    
    if future_data.shape[0] == 1:
        future_data = future_data.reshape(1, 1)
    else:
        future_data = future_data.reshape(future_data.shape[0], 1)
        
    future_data = scaler.inverse_transform(future_data)
    return future_data

# Streamlit app
def main():
    st.title('ARIMA Data Forecasting')

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['time_interval'] = pd.to_datetime(df['time_interval'])
        df.set_index('time_interval', inplace=True)

        X, y, scaler = preprocess_data(df)
        model = build_model(X, y)

        # Forecast data for 1 day (24 steps)
        last_x = X[-1]
        future_data = forecast_data(model, last_x, scaler, 24)
        forecast_timestamps = pd.date_range(start=df.index[-1], periods=len(future_data) + 1, freq='H')[1:]
        
        # Create DataFrame for forecasted data
        forecast_df = pd.DataFrame({'Delivery Interval': forecast_timestamps, 'Forecasted Value': future_data[:, 0]})
        forecast_df.set_index('Delivery Interval', inplace=True)

        # Display forecasted data
        st.subheader('Forecasted Data')
        st.write(forecast_df)

        # Plot forecasted data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=future_data[:, 0], name='Forecasted Data'))
        fig.update_layout(title='1-Day Forecast using ARIMA', xaxis_title='Delivery Interval', yaxis_title='Average LMP')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
