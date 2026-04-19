import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

print("Training LSTM Forecasting Model...")

df = pd.read_csv('data/singrauli_sensor_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['zone', 'timestamp'])

zone_df = df[df['zone'] == 'Zone_1A']['CAR'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(zone_df)

LOOKBACK  = 48
FORECAST  = 24

def create_sequences(data, lookback, forecast):
    X, y = [], []
    for i in range(len(data) - lookback - forecast):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+forecast])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, LOOKBACK, FORECAST)

split = int(len(X) * 0.8)
X_train = torch.FloatTensor(X[:split])
y_train = torch.FloatTensor(y[:split])
X_test  = torch.FloatTensor(X[split:])
y_test  = torch.FloatTensor(y[split:])

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32, shuffle=True)

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, output_size=24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers, batch_first=True,
            dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model     = LSTMForecaster()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 30
print(f"Training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.6f}")

model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, y_test.squeeze(-1))
    print(f"\nTest Loss (MSE): {test_loss.item():.6f}")

os.makedirs('ml', exist_ok=True)
torch.save(model.state_dict(), 'ml/lstm_model.pt')
pickle.dump(scaler, open('ml/lstm_scaler.pkl', 'wb'))

config = {
    'input_size':  1,
    'hidden_size': 64,
    'num_layers':  2,
    'output_size': 24,
    'lookback':    LOOKBACK,
    'forecast':    FORECAST
}
pickle.dump(config, open('ml/lstm_config.pkl', 'wb'))
print("Trained....")