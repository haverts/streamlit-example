import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os
import glob

OutputPath = r'C:\Users\ReymondJardin\Desktop'

df = pd.read_csv(r'C:\Users\ReymondJardin\Desktop\dipcef.csv', index_col='time_interval', parse_dates=True)
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
df = df.resample('1H').mean()
df

train_data = df[:-4392]
test_data = df[-4392:]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(train_scaled.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

# Compile the LSTM model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the LSTM model to the training data
lstm_model.fit(train_scaled, train_scaled, epochs=100, batch_size=32)

# Make predictions using the LSTM model
lstm_predictions = lstm_model.predict(test_scaled)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Define the SARIMA model
sarima_model = SARIMAX(train_data, order=(2, 1, 2), seasonal_order=(0, 1, 1, 4392))

# Fit the SARIMA model to the training data
sarima_fit = sarima_model.fit()

# Make predictions using the SARIMA model
sarima_predictions = sarima_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)

test_data.index = pd.to_datetime(test_data.index, errors='coerce')
test_data = test_data.shift(freq='4392H')

lstm_predictions = pd.DataFrame(lstm_predictions, index=test_data.index)
lstm_predictions = lstm_predictions.rename(columns={0: 'avg_lmp'})

import plotly.graph_objects as go
import pandas as pd

trace1 = go.Scatter(x=df.index, y=df.avg_lmp, name='Actual')
trace2 = go.Scatter(x=list(lstm_predictions.index), y=lstm_predictions.avg_lmp, name='LSTM')
trace3 = go.Scatter(x=list(test_data.index), y=sarima_predictions, name='SARIMAX')


layout = go.Layout(title='Actual vs Forecast LMP using LSTM and SARIMAX')
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()

fig.write_html(fr'{OutputPath}\forecast.html',  auto_open=True)