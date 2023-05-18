import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Perform necessary data preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler()
    data['ScaledValue'] = scaler.fit_transform(data['avg_lmp'].values.reshape(-1, 1))
    return data, scaler

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(None, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sarima_model():
    model = SARIMAX(data['avg_lmp'], order=(1, 0, 0), seasonal_order=(1, 1, 0, 4392))
    model = model.fit()
    return model

# Define the forecasting function for LSTM
def lstm_forecast(data, scaler):
    lstm_model = create_lstm_model()
    X = np.array(data['ScaledValue']).reshape(-1, 1)
    lstm_model.fit(X[:-4392], data['ScaledValue'].values[4392:], epochs=10, batch_size=32, verbose=0)
    forecast = lstm_model.predict(X[-4392:])
    forecast = scaler.inverse_transform(forecast)
    return forecast.flatten()

# Define the forecasting function for SARIMA
def sarima_forecast(data):
    sarima_model = create_sarima_model()
    sarima_model_fit = sarima_model.fit()
    forecast = sarima_model_fit.forecast(steps=4392)
    return forecast

# Streamlit app
def main():
    df = pd.read_csv(r'dipcef.csv', index_col='time_interval', parse_dates=True)
    #path = r'/content/sample_data'
    #all_files = glob.glob(os.path.join(path, "dipcef_*.csv"))

    #df = pd.concat((pd.read_csv(f, parse_dates=True) for f in all_files))

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    #df['time_interval'] = pd.to_datetime(df['time_interval'])
    #df = df.drop(columns=['MKT_TYPE', 'COMMODITY_TYPE','RESOURCE_NAME', 'RESOURCE_TYPE' , 'REGION_NAME' ,'RUN_TIME'])
    print('Shape of Data' , df.shape)
    df = df.dropna()
    df.info()

    import datetime
    #df.index = df.index.date
    #df = df.groupby(df.index)['avg_lmp'].agg(avg_lmp=('mean'))
    # Resample the data to hourly intervals
    data = df.resample('1H').mean()
    data, scaler = preprocess_data(data)
    
    # Set up the sidebar with model selection
    models = ['LSTM', 'SARIMA']
    model_selection = st.sidebar.selectbox('Select Model', models)
    
    # Generate the forecast based on the selected model
    if model_selection == 'LSTM':
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
            
              # Plot forecast
            trace1 = go.Scatter(x=df.index, y=df.avg_lmp, name='Actual')
            trace2 = go.Scatter(x=list(lstm_predictions.index), y=lstm_predictions.avg_lmp, name='LSTM')
            layout = go.Layout(title='Actual vs Forecast LMP')
            fig = go.Figure(data=[trace1, trace2], layout=layout)
            fig.show()
            
    elif model_selection == 'SARIMA':
        data, scaler = preprocess_data(data)
        forecast = sarima_forecast(data)
    
    # Display the hourly forecast
    st.write('Hourly Forecast:')
    st.write(pd.DataFrame({'Forecast': forecast}, index=data.tail(4392).index))

if __name__ == '__main__':
    main()
