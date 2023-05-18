import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['time_interval'] = pd.to_datetime(df['time_interval'])
    df.set_index('time_interval', inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    return scaled_data, scaler

# Create the LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Generate the forecast
def generate_forecast(data, scaler):
    model = create_lstm_model()
    model.fit(data[:-12], data[12:], epochs=50, batch_size=32)
    last_60 = data[-60:]
    forecast = []
    for _ in range(12):
        prediction = model.predict(last_60.reshape(1, 60, 1))
        forecast.append(prediction[0])
        last_60 = np.roll(last_60, -1)
        last_60[-1] = prediction[0]
    forecast = scaler.inverse_transform(forecast)
    return forecast

# Main function
def main():
    st.title("LSTM Forecasting")
    file = st.file_uploader("Upload CSV file", type="csv")
    if file is not None:
        data, scaler = load_data(file.name)
        forecast = generate_forecast(data, scaler)
        
        # Create hourly timestamps for x-axis
        start_time = df.index[-1] + pd.Timedelta(minutes=5)
        hourly_timestamps = pd.date_range(start=start_time, periods=12, freq='H')
        
        # Plot the forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=scaler.inverse_transform(data),
                                 name='Actual Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=hourly_timestamps, y=forecast,
                                 name='Forecast', line=dict(color='orange')))
        fig.update_layout(xaxis=dict(tickformat='%H:%M', title='Delivery_Interval'),
                          yaxis=dict(title='avg_lmp'))
        st.plotly_chart(fig)

# Run the app
if __name__ == '__main__':
    main()
