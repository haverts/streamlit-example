import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


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
    model.fit(X, y, epochs=2, batch_size=32)
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
def build_arima_model(df):
    # Convert to hourly interval
    df_hourly = df.resample('1H').mean().interpolate()

    # Train-test split
    train_size = int(len(df_hourly) * 0.8)
    train_data = df_hourly[:train_size]
    test_data = df_hourly[train_size:]

    # Train the ARIMA model
    model = ARIMA(train_data, order=(2, 1, 2))  # Specify the order (p, d, q)
    model_fit = model.fit()

    return model_fit, test_data

# Function to forecast data using ARIMA model
def forecast_arima_data(model, test_data):
    forecast = model.forecast(steps=len(test_data))[0]
    return forecast

    for i in range(24):
        prediction = model.forecast(steps=2160)[0]  # Use ARIMA model for forecasting
        future_data.append(prediction)
        last_x = np.concatenate((last_x[1:], prediction), axis=0)

    future_data = np.array(future_data)
    future_data = scaler.inverse_transform(future_data)
    return future_data



# Streamlit app
# Streamlit app
def main():
    st.title('LSTM and ARIMA Data Forecasting')

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['time_interval'] = pd.to_datetime(df['time_interval'])
        df.set_index('time_interval', inplace=True)

        X, y, scaler = preprocess_data(df)
        model = build_model(X, y)

        # Forecast data using LSTM model
        last_x = X[-1]
        future_data_lstm = forecast_data(model, last_x, scaler)
        forecast_timestamps = pd.date_range(start=df.index[-1], periods=len(future_data_lstm) + 1, freq='H')[1:]

        # Train and forecast using ARIMA model
        arima_model, test_data = build_arima_model(df)
        future_data_arima = forecast_arima_data(arima_model, test_data)
        
        # Create DataFrame for LSTM forecasted data
        forecast_df_lstm = pd.DataFrame({'Delivery Interval': forecast_timestamps, 'Forecasted Value (LSTM)': future_data_lstm[:, 0]})
        forecast_df_lstm.set_index('Delivery Interval', inplace=True)
        
         # Create DataFrame for ARIMA forecasted data
        forecast_df_arima = pd.DataFrame({'Delivery Interval': forecast_timestamps, 'Forecasted Value (ARIMA)': future_data_arima})
        forecast_df_arima.set_index('Delivery Interval', inplace=True)

        # Display LSTM forecasted data
        st.subheader('LSTM Forecasted Data')
        forecast_df_lstm_placeholder = st.empty()
        forecast_df_lstm_placeholder.write(forecast_df_lstm)
       

        
        # Display ARIMA forecasted data
        st.subheader('ARIMA Forecasted Data')
        forecast_df_arima_placeholder = st.empty()
        forecast_df_arima_placeholder.write(forecast_df_arima)



        # Plot forecasted data
        future_data_arima_y = future_data_arima[:, 0].tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=future_data_lstm[:, 0], name='Forecasted Data (LSTM)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=forecast_df_arima, name='Forecasted Data (ARIMA)', line=dict(color='red', dash='dot')))
        fig.update_layout(
            title='1-Day Forecast using LSTM and ARIMA',
            xaxis_title='Delivery Interval',
            yaxis_title='Average LMP',
            legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0.5)'),
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background color to transparent
            font=dict(color='black')
        )
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')  # Customize grid color
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')  # Customize grid color
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
