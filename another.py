import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pmdarima.arima import auto_arima

# Function to simplify data to hourly intervals
def simplify_to_hourly(data):
    data = data.interpolate(method='linear')
    hourly_data = data.resample('1H').mean().ffill()
    return hourly_data

# Function to create LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(24, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to create SARIMA model
def create_sarima_model(data):
    model = auto_arima(data, seasonal=True, m=12)
    return model

# Function to plot the forecast
def plot_forecast(data, forecast):
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['avg_lmp'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Prediction'], name='Forecast'))
    fig.update_layout(height=600, width=800, title_text='Forecast')
    st.plotly_chart(fig)

# Streamlit app
def main():
    # Page title
    st.title('1 Year Forecast Simplification')

    # Data upload
    st.header('Data Upload')
    uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data['time_interval'] = pd.to_datetime(data['time_interval'])
        data.set_index('time_interval', inplace=True)
        st.success('Data uploaded successfully!')
        st.subheader('Data Preview in 5min interval')
        st.write(data.head())

        # Data simplification
        st.header('Market Energy Price')
        simplified_data = simplify_to_hourly(data)
        st.subheader('Simplified Data Preview in Hourly')
        st.write(simplified_data.head())

        # Model selection
        st.header('Model Selection')
        model = st.selectbox('Select Model', ['LSTM', 'SARIMA'])

        # Model training and forecasting
        st.header('Model Forecast')
        if model == 'LSTM':
            # Data scaling
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(simplified_data)

            # Prepare LSTM training data
            train_data = scaled_data[:int(0.9*len(scaled_data))]
            x_train = []
            y_train = []
            for i in range(24, len(train_data)):
                x_train.append(train_data[i-24:i, 0])
                y_train.append(train_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Create and train LSTM model
            lstm_model = create_lstm_model()
            lstm_model.fit(x_train, y_train, epochs=10, batch_size=32)

            # Forecasting
            test_data = scaled_data[int(0.9*len(scaled_data)):]
            x_test = []
            y_test = simplified_data[int(0.9*len(scaled_data)):]
            for i in range(24, len(test_data)):
                x_test.append(test_data[i-24:i, 0])
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            predicted_data = lstm_model.predict(x_test)
            predicted_data = scaler.inverse_transform(predicted_data)

        elif model == 'SARIMA':
            # Create and fit SARIMA model
            sarima_model = create_sarima_model(simplified_data)
            sarima_model.fit(simplified_data)

            # Forecasting
            predicted_data = sarima_model.predict(n_periods=len(simplified_data))
            predicted_data = pd.Series(predicted_data, index=simplified_data.index)

        # Plot forecast
        forecast = simplified_data.copy()
        forecast['Prediction'] = predicted_data
        plot_forecast(simplified_data, forecast)

if __name__ == '__main__':
    main()
