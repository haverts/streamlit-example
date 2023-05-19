import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
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
    lookback = 24*7
    X = []
    y = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    # Reshape X for LSTM input shape (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Function to build and train the LSTM model
def build_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)
    return model

# Function to forecast data
def forecast_data(model, last_x, scaler):
    future_data = []

    for i in range(7*24):
        prediction = model.predict(np.array([last_x]))
        future_data.append(prediction[0])
        last_x = np.concatenate((last_x[1:], prediction), axis=0)

    future_data = np.array(future_data)
    future_data = scaler.inverse_transform(future_data)
    return future_data

def build_arima_model(data):
    model = ARIMA(data, order=(2, 1, 0))  # Define the ARIMA order
    model_fit = model.fit()
    return model_fit

def build_model1(X, y):
    model_fit = build_arima_model(y)  # Use ARIMA model instead of LSTM
    return model_fit

def forecast_arima_data(model, last_x, scaler):
    future_data = []

    for i in range(7*24):
        prediction = model.forecast(steps=1)[0]  # Use ARIMA model for forecasting
        future_data.append(prediction)
        last_x = np.concatenate((last_x[1:], prediction), axis=0)

    future_data = np.array(future_data)
    future_data = scaler.inverse_transform(future_data)
    return future_data



# Streamlit app
def main():
    st.title('LSTM Data Forecasting')
    # Model selection
    model_type = st.sidebar.selectbox("Select Model", ("LSTM", "ARIMA"))

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        if model_type == "LSTM":
            model = build_model(X, y)
            last_x = scaled_data[-1]
            future_data = forecast_data(model, last_x, scaler)

            df = pd.read_csv(uploaded_file)
            df['time_interval'] = pd.to_datetime(df['time_interval'])
            df.set_index('time_interval', inplace=True)

            X, y, scaler = preprocess_data(df)
            model = build_model(X, y)

            # Forecast data for 1 day
            last_x = X[-1]
            future_data = forecast_data(model, last_x, scaler)
            forecast_timestamps = pd.date_range(start=df.index[-1], periods=len(future_data) + 1, freq='H')[1:]

            # Create DataFrame for forecasted data
            forecast_df = pd.DataFrame({'Delivery Interval': forecast_timestamps, 'Forecasted Value': future_data[:, 0]})
            forecast_df.set_index('Delivery Interval', inplace=True)

            # Display forecasted data
            st.subheader('Forecasted LSTM Data')
            st.write(forecast_df)

            # Plot forecasted data
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_timestamps, y=future_data[:, 0], name='Forecasted LSTM Data'))
            fig.update_layout(title='1-Day Forecast using LSTM', xaxis_title='Delivery Interval', yaxis_title='Average LMP')
            st.plotly_chart(fig)
        else:
            df = pd.read_csv(uploaded_file)
            df['time_interval'] = pd.to_datetime(df['time_interval'])
            df.set_index('time_interval', inplace=True)

            scaled_data, original_data, scaler = preprocess_data(df)
            model1 = build_model1(scaled_data, original_data)

            # ARIMA forecast
            model_arima = build_arima_model(original_data)
            last_x_arima = original_data.values[-1]
            future_data_arima = forecast_arima_data(model_arima, last_x_arima, scaler)
      
            forecast_df_arima = pd.DataFrame({'Delivery Interval': forecast_timestamps, 'Forecasted ARIMA Value': future_arima_data[:, 0]})
            forecast_df_arima.set_index('Delivery Interval', inplace=True)

            # Display forecasted data
            st.subheader('Forecasted ARIMA Data')
            st.write(forecast_df_arima)

            # Plot forecasted data
            # Plot forecasted data
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_timestamps, y=future_arima_data[:, 0], name='Forecasted ARIMA Data'))
            fig.update_layout(title='1-Day Forecast using LSTM', xaxis_title='Delivery Interval', yaxis_title='Average LMP')
            st.plotly_chart(fig)
if __name__ == '__main__':
    main()
