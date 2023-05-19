import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Function to preprocess the data
def preprocess_data(df):
    # Convert to hourly interval
    df_hourly = df.resample('1H').mean().interpolate()

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_hourly)

    # Prepare the data for LSTM model
    # User input for forecasting steps
    lookback = 24 * 7
    X = []
    y = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    # Reshape X for LSTM input shape (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Function to build and train the LSTM model
def build_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)
    return model

# Function to build and train the ARIMA model
def build_arima_model(data):
    model = ARIMA(data, order=(1, 0, 0))
    model_fit = model.fit()
    return model_fit

# Function to forecast data using LSTM
def forecast_lstm(model, last_x, scaler):
    future_data = []

    for _ in range(7 * 24):
        prediction = model.predict(np.array([last_x]))
        future_data.append(prediction[0])
        last_x = np.concatenate((last_x[1:], prediction), axis=0)

    future_data = np.array(future_data)
    future_data = scaler.inverse_transform(future_data)
    return future_data

# Function to forecast data using ARIMA
def forecast_arima(model, steps):
    forecast_data = model.forecast(steps=steps)[0]
    return forecast_data

# Streamlit app
def main():
    st.title('Time Series Forecasting')

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['time_interval'] = pd.to_datetime(df['time_interval'])
        df.set_index('time_interval', inplace=True)

        X, y, scaler = preprocess_data(df)

        # Model selection
        model_type = st.selectbox("Select model", options=["LSTM", "ARIMA"])

        if model_type == "LSTM":
            model = build_lstm_model(X, y)

            # Forecast data for 1 day
            last_x = X[-1]
            future_data = forecast_lstm(model, last_x, scaler)
        else:
            model = build_arima_model(df['value'])

            # Forecast data for 1 day
            steps = 24
            future_data = forecast_arima(model, steps)

        forecast_timestamps = pd.date_range(start=df.index[-1], periods=len(future_data), freq='H')[::-1]

        # Create DataFrame for forecasted data
        forecast_df = pd.DataFrame({'Delivery Interval': forecast_timestamps, 'Forecasted Value': future_data})
        forecast_df.set_index('Delivery Interval', inplace=True)

        # Display forecasted data
        st.subheader('Forecasted Data')
        st.write(forecast_df)

        # Plot forecasted data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=future_data, name='Forecasted Data'))
        fig.update_layout(title='1-Day Forecast', xaxis_title='Delivery Interval', yaxis_title='Value')
        st.plotly_chart(fig)


if __name__ == '__main__':
    main()
