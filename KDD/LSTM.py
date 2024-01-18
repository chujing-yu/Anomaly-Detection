import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Download time series data using yfinance
data = yf.download('AAPL', start='2018-01-01', end='2023-06-30')
# print(data)


# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('AAPL Stock Price')
plt.xticks(rotation=45)
plt.grid(True)
# plt.show()


# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)


# Smooth the time series data using a moving average
window_size = 7
data['Smoothed'] = data['Close'].rolling(window_size).mean()
# Plot the smoothed data
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Original')
plt.plot(data['Smoothed'], label='Smoothed')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('AAPL Stock Price (Smoothed)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
# plt.show()


# Calculate z-scores for each data point
z_scores = (data['Close'] - data['Close'].mean()) / data['Close'].std()
# Define a threshold for outlier detection
threshold = 3
# Identify outliers
outliers = data[np.abs(z_scores) > threshold]
# Remove outliers from the data
data = data.drop(outliers.index)
# Plot the data without outliers
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('AAPL Stock Price (Without Outliers)')
plt.xticks(rotation=45)
plt.grid(True)
# plt.show()


# Calculate z-scores for each data point
z_scores = (data['Close'] - data['Close'].mean()) / data['Close'].std()
# Plot the z-scores
plt.figure(figsize=(12, 6))
plt.plot(z_scores)
plt.xlabel('Date')
plt.ylabel('Z-Score')
plt.title('Z-Scores for AAPL Stock Price')
plt.xticks(rotation=45)
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.axhline(y=-threshold, color='r', linestyle='--')
plt.legend()
plt.grid(True)

plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare the data for LSTM Autoencoder
X = data['Close'].values.reshape(-1, 1)

# Normalize the data
X_normalized = (X - X.min()) / (X.max() - X.min())

# Train the LSTM Autoencoder model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_normalized, X_normalized, epochs=10, batch_size=32)

# Reconstruct the input sequence
X_reconstructed = model.predict(X_normalized)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.abs(X_normalized - X_reconstructed), axis=1)

# Plot the reconstruction error
plt.figure(figsize=(12, 6))
plt.plot(reconstruction_error)
plt.xlabel('Date')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error (LSTM Autoencoder)')
plt.xticks(rotation=45)
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.grid(True)

plt.show()