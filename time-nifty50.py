import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


stock_symbol = '^NSEI'
start_date = '2020-01-01'
end_date = '2024-11-01'


data = yf.download(stock_symbol, start=start_date, end=end_date)

close_prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)
def create_lstm_data(data, time_steps=1):
  x, y = [], []
  for i in range(len(data) - time_steps):
    x.append(data[i:(i + time_steps), 0])
    y.append(data[i + time_steps, 0])
  return np.array(x), np.array(y)

time_steps = 10
x, y = create_lstm_data(close_prices_scaled, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=
(x.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=50, batch_size=32,)
predicted_prices = model.predict(x)


predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))
print(f"Predicted Price: {predicted_prices.flatten()[0]}")
mse = mean_squared_error(actual_prices, predicted_prices)
print(f'Mean Squared Error: {mse}')


plt.figure(figsize=(14, 7))
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('Nifty 50 Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
