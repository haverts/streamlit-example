import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go


# Main function
def main():
    st.title("LSTM Forecasting")
    file = st.file_uploader("Upload CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        df['TIME_INTERVAL'] = pd.to_datetime(df['TIME_INTERVAL'])
        df.set_index('TIME_INTERVAL', inplace=True)

        # Convert to hourly interval
        df_hourly = df.resample('1H').mean().interpolate()

        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_hourly)
        st.success('Data uploaded successfully!')
        st.subheader('Data Preview in 5min interval')
        st.write(df.head())

        # Step 1: Load and preprocess the data
        # Assuming you have a CSV file named 'data.csv' with a 'timestamp' and 'value' column
        df = pd.read_csv('dipcef2.csv')
        df['TIME_INTERVAL'] = pd.to_datetime(df['TIME_INTERVAL'])
        df.set_index('TIME_INTERVAL', inplace=True)

        # Convert to hourly interval
        df_hourly = df.resample('1H').mean().interpolate()

        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_hourly)

        # Step 2: Prepare the data for LSTM model
        lookback = 24  # Number of previous hours to use for prediction

        X = []
        y = []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i])

        X = np.array(X)
        y = np.array(y)

        # Reshape X for LSTM input shape (samples, time steps, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Step 3: Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Step 4: Train the LSTM model
        model.fit(X, y, epochs=5, batch_size=32)

        # Step 5: Forecast the next 1 year of data
        future_data = []
        last_x = scaled_data[-lookback:]
        for i in range(365 * 24):
            prediction = model.predict(np.array([last_x]))
            future_data.append(prediction[0])
            last_x = np.concatenate((last_x[1:], prediction), axis=0)

        # Step 6: Inverse scale the forecasted data
        future_data = np.array(future_data)
        future_data = scaler.inverse_transform(future_data)

        # Step 7: Create timestamps for the forecasted data
        start_timestamp = df_hourly.index[-1] + pd.DateOffset(hours=1)
        forecast_timestamps = pd.date_range(start=start_timestamp, periods=len(future_data), freq='H')

        # Step 8: Plot the forecasted data using Plotly
        actual_data = df_hourly['avg_lmp']
        forecasted_data = pd.Series(future_data[:, 0], index=forecast_timestamps)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data, name='Actual Data'))
        fig.add_trace(go.Scatter(x=forecasted_data.index, y=forecasted_data, name='Forecasted Data'))
        fig.update_layout(title='1-Year Forecast using LSTM', xaxis_title='Timestamp', yaxis_title='Value')
        fig.show()

            
# Run the app
if __name__ == '__main__':
    main()
