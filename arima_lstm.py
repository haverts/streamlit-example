import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Assuming your data has a 'timestamp' column
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')

    # Step 2: Preprocess the Data
    # Convert to hourly intervals if necessary
    # Assuming your data is already at hourly intervals

    # Split into training and testing sets
    train_data = data[:-24]  # Use the first 6 months for training (assuming 24 hours per day)
    test_data = data[-24:]  # Use the last day for testing (24 hours)

    # Step 3: ARIMA Model
    # Fit ARIMA model to training data
    arima_model = ARIMA(train_data, order=(2, 1, 0))  # ARIMA(2, 1, 0) as an example
    arima_model_fit = arima_model.fit()

    # Forecast one day of data
    arima_forecast = arima_model_fit.forecast(steps=24)

    # Plot ARIMA forecast
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data.values, label='Actual')
    plt.plot(test_data.index, arima_forecast, label='ARIMA Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('ARIMA Forecast')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

    # Create a DataFrame for future ARIMA forecast
    future_arima_df = pd.DataFrame(arima_forecast, columns=['ARIMA Forecast'])
    future_arima_df.index = pd.date_range(start=data.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')

    # Step 4: LSTM Model
    # Normalize the data
    max_value = train_data.max().values
    min_value = train_data.min().values
    normalized_data = (train_data - min_value) / (max_value - min_value)

    # Reshape the data for LSTM
    X_train = []
    y_train = []
    for i in range(24, len(normalized_data)):
        X_train.append(normalized_data[i - 24:i, 0])
        y_train.append(normalized_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the data for prediction
    last_day = normalized_data[-24:].values
    last_day = np.reshape(last_day, (1, last_day.shape[0], 1))

    # Build LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(24, 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the LSTM model
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Forecast one day of data
    lstm_forecast = lstm_model.predict(last_day)
    lstm_forecast = lstm_forecast * (max_value - min_value) + min_value

    # Plot LSTM forecast
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data.values, label='Actual')
    plt.plot(test_data.index, lstm_forecast.flatten(), label='LSTM Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('LSTM Forecast')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

    # Create a DataFrame for future LSTM forecast
    future_lstm_df = pd.DataFrame(lstm_forecast.flatten(), columns=['LSTM Forecast'])
    future_lstm_df.index = pd.date_range(start=data.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')

    # Step 5: Visualizing the Results
    st.subheader('ARIMA Forecast')
    st.write(future_arima_df)

    st.subheader('LSTM Forecast')
    st.write(future_lstm_df)
