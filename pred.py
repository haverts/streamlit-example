import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
    lookback = 12
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

# Function to build and train the ARIMA model
def build_arima_model(data):
    model1 = ARIMA(data, order=(1, 0, 0))
    model1_fit = model1.fit()
    return model1_fit

# Function to forecast data using the ARIMA model
def forecast_arima(model1, steps):
    future_data = model1.forecast(steps=steps)[0]
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
        
        model1 = build_arima_model(df['avg_lmp'])

        # Forecast data for 1 day
        steps = 24
        last_x1 = X[-1]
        future_data_arima = forecast_arima(model1, steps)
        forecast_timestamps_arima = pd.date_range(start=df.index[-1], periods=len(future_data) + 1, freq='H')[1:]
        
                
         # Create DataFrame for forecasted data
        forecast_arima_df = pd.DataFrame({'Delivery Interval': forecast_timestamps_arima, 'Forecasted Value': future_data_arima[:, 0]})
        forecast_arima_df.set_index('Delivery Interval', inplace=True)

         # Display forecasted data
        st.subheader('Forecasted Data')
        st.write(forecast_arima_df)

        # Plot forecasted data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=future_data[:, 0], name='Forecasted Data'))
        fig.update_layout(title='1-Day Forecast using LSTM', xaxis_title='Delivery Interval', yaxis_title='Average LMP')
        st.plotly_chart(fig)
        

        trace2 = go.Scatter(x=list(forecast_timestamps), y=future_data, name='LSTM')
        trace3 = go.Scatter(x=list(test_data.index), y=sarima_predictions, name='ARIMA')


        layout = go.Layout(title='Actual vs Forecast LMP using LSTM and SARIMAX')
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        fig.show()


if __name__ == '__main__':
    main()
