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
            
            train_data = data[:-4392]
            test_data = data[-4392:]
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(simplified_data)

            # Prepare LSTM training data
            # Data scaling
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            # Prepare training data
            train_data = scaled_data[:-8760]  # Use all but the last year for training
            x_train, y_train = [], []
            for i in range(8760, len(train_data)):
                x_train.append(train_data[i-8760:i, 0])
                y_train.append(train_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Create and train LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=10, batch_size=32)
 
            # Forecasting
            last_year_data = scaled_data[-8760:]  # Use the last year for forecasting
            x_test = []
            for i in range(8760, len(last_year_data)):
                x_test.append(last_year_data[i-8760:i, 0])
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            predicted_data = model.predict(x_test)
            predicted_data = scaler.inverse_transform(predicted_data)
            
        elif model == 'SARIMA':
    # Plot forecast
    trace1 = go.Scatter(x=df.index, y=df.avg_lmp, name='Actual')
    trace2 = go.Scatter(x=list(lstm_predictions.index), y=lstm_predictions.avg_lmp, name='LSTM')
    layout = go.Layout(title='Actual vs Forecast LMP')
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.show()


if __name__ == '__main__':
    main()
