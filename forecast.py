import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Set Streamlit app title
st.title("Time Series Forecasting with LSTM")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.subheader("Uploaded Data")
    st.dataframe(df)

    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df["Value"].values.reshape(-1, 1))

    # Configure model parameters
    look_back = 12  # Number of previous time steps to use as input for predicting the next time step
    units = 50  # Number of LSTM units
    epochs = 100  # Number of training epochs

    # Prepare the training data
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    x_train, y_train = [], []
    for i in range(look_back, len(train_data)):
        x_train.append(train_data[i - look_back:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2)

    # Prepare the test data
    test_data = scaled_data[train_size - look_back:]
    x_test, y_test = [], []
    for i in range(look_back, len(test_data)):
        x_test.append(test_data[i - look_back:i, 0])
        y_test.append(test_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Generate predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Create hourly and daily time series
    hourly_predictions = pd.Series(predictions.flatten(), index=df.index[train_size:]).resample("H").mean()
    daily_predictions = pd.Series(predictions.flatten(), index=df.index[train_size:]).resample("D").mean()

    # Display the forecasted results
    st.subheader("Hourly Time Series Forecast")
    st.line_chart(hourly_predictions)

    st.subheader("Daily Time Series Forecast")
    st.line_chart(daily_predictions)
