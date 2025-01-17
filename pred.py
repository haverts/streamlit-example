import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import statsmodels.api as sm

# Function to preprocess the data
def preprocess_data(df):
    # Convert to hourly interval
    df_hourly = df.resample('1H').mean().interpolate()

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_hourly)

    # Prepare the data for LSTM model
    # User input for forecasting steps
    lookback = 2160
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

    for i in range(24):
        prediction = model.predict(np.array([last_x]))
        future_data.append(prediction[0])
        last_x = np.concatenate((last_x[1:], prediction), axis=0)

    future_data = np.array(future_data)
    future_data = scaler.inverse_transform(future_data)
    return future_data


# Streamlit app
def main():
    st.title('LSTM Data Forecasting')

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
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
        st.subheader('Forecasted Data')
        st.write(forecast_df)
        
        # Plot forecasted data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=future_data[:, 0], name='Forecasted Data'))
        fig.update_layout(title='1-Day Forecast using LSTM', xaxis_title='Delivery Interval', yaxis_title='Average LMP')
        st.plotly_chart(fig)
        
                # Preprocess the data
        series = preprocess_data(df)

        # Fit an ARIMA model
        model = sm.tsa.ARIMA(series, order=(1, 0, 0))  # Example order (p, d, q)
        results = model.fit()
        
        # Forecast data for 1 day
        forecast_steps = 24  # Number of steps to forecast
        forecast = results.forecast(steps=forecast_steps)
        

        # Convert the forecasted values to a numpy array
        future_data_arima = np.array(forecast[0])
        
                # Inverse transform the forecasted data
        future_data_arima = scaler.inverse_transform(future_data_arima.reshape(-1, 1))
        # Display forecasted data
        st.subheader('ARIMA Forecasted Data')
                # Create DataFrame for forecasted data
        forecast_df_arima = pd.DataFrame({'Delivery Interval': forecast_timestamps, 'Forecasted Value': future_data_arima[:, 0]})
        forecast_df_arima.set_index('Delivery Interval', inplace=True)

        # Plot forecasted data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=forecast_df_arima[:, 0], name='Forecasted Data'))
        fig.update_layout(title='1-Day Forecast using LSTM', xaxis_title='Delivery Interval', yaxis_title='Average LMP')
        st.plotly_chart(fig)



if __name__ == '__main__':
    main()
