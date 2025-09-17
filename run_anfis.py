import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.ANFIS.ANFIS import ANFIS

# 1. Load and Clean Data
try:
    # Load the dataset from the uploaded file
    df = pd.read_csv('stock_data.csv')

    # Clean the data: remove trailing empty rows and select the 'Close' column
    df.dropna(how='all', inplace=True)
    stock_price = df['Close'].values.astype(np.float32)
    print(f"Successfully loaded and cleaned data. Shape: {stock_price.shape}")

except FileNotFoundError:
    print("Error: 'stock_data.csv' not found. Please ensure the file is in the correct directory.")
    # As a fallback, generate synthetic data so the rest of the script can run
    stock_price = np.sin(np.linspace(0, 400, 6636)) * 50 + 100

# 2. Preprocess Data
# Create a MinMaxScaler to normalize data to a 0-1 range
scaler = MinMaxScaler(feature_range=(0, 1))
stock_price_scaled = scaler.fit_transform(stock_price.reshape(-1, 1)).flatten()

# 3. Feature Engineering (Temporal Windowing)
X, y = [], []
lags = 5  # Use 5 previous days to predict the next
for i in range(lags, len(stock_price_scaled)):
    X.append(stock_price_scaled[i - lags:i])
    y.append(stock_price_scaled[i])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# 4. Data Splitting
# Use 80% of the data for training, 20% for testing
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Convert to PyTorch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(f"Data prepared. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# 6. Training
model = ANFIS(input_dim=lags, num_mfs=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0105)
criterion = nn.MSELoss(reduction='mean')

epochs = 100
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_output = model(X_train)
    train_loss = torch.sqrt(criterion(train_output, y_train))
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())

    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = torch.sqrt(criterion(test_output, y_test))
        test_losses.append(test_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train RMSE: {train_loss.item():.4f}, Test RMSE: {test_loss.item():.4f}')
