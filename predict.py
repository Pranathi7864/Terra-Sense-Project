import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
xgb_model = pickle.load(open('ml/car_model.pkl',       'rb'))
iso_model  = pickle.load(open('ml/isolation_model.pkl', 'rb'))
scaler     = pickle.load(open('ml/lstm_scaler.pkl',     'rb'))
config     = pickle.load(open('ml/lstm_config.pkl',     'rb'))

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

lstm_model = LSTMForecaster()
lstm_model.load_state_dict(torch.load('ml/lstm_model.pt'))
lstm_model.eval()

def full_predict(sensor_data: dict, car_history: list = None):
    """
    sensor_data  : dict of current sensor readings
    car_history  : list of last 48 CAR values (for LSTM)

    Returns full prediction from all 3 models
    """

    features = ['soil_moisture', 'co2_ppm', 'temperature',
                 'humidity', 'vibration', 'ndvi',
                 'rainfall_mm', 'mining_activity']

    X = pd.DataFrame([sensor_data])[features].fillna(0)

    # XGBoost — Current CAR
    car_now = float(xgb_model.predict(X)[0])
    car_now = np.clip(car_now, 0, 1)

    # Isolation Forest — Anomaly
    anomaly_flag = iso_model.predict(X)[0]
    is_anomaly   = anomaly_flag == -1

    #  LSTM — Future CAR 
    future_car = None
    hours_to_critical = None

    if car_history and len(car_history) >= 48:
        history = np.array(car_history[-48:]).reshape(-1, 1)
        scaled  = scaler.transform(history)
        tensor  = torch.FloatTensor(scaled).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = lstm_model(tensor).numpy()[0]

        future_car = scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)).flatten().tolist()

        for i, val in enumerate(future_car):
            if val < 0.25:
                hours_to_critical = i * 0.5 
                break

    if is_anomaly and car_now < 0.40:
        risk = "CRITICAL"
        alert = True
    elif car_now >= 0.65:
        risk  = "STABLE"
        alert = False
    elif car_now >= 0.40:
        risk  = "WATCH"
        alert = False
    elif car_now >= 0.25:
        risk  = "WARNING"
        alert = True
    else:
        risk  = "CRITICAL"
        alert = True

    if risk == "STABLE":
        recommendation = "No action needed. Continue monitoring."
    elif risk == "WATCH":
        recommendation = "Begin afforestation planning. Monitor closely."
    elif risk == "WARNING":
        recommendation = (
            "Plant Vetiver grass immediately. "
            "Improve drainage. Schedule backfilling."
        )
    else:
        recommendation = (
            "IMMEDIATE: Reroute defense convoys. "
            "Deploy geo-textile reinforcement. "
            "Emergency Vetiver grass planting across affected zone. "
            "Backfill subsurface cavities."
        )

    return {
        "car_current":        round(car_now, 4),
        "risk":               risk,
        "alert":              alert,
        "is_anomaly":         is_anomaly,
        "future_car_12hrs":   future_car,
        "hours_to_critical":  hours_to_critical,
        "recommendation":     recommendation
    }


if __name__ == "__main__":
    
    test_sensor = {
        'soil_moisture':  18.0,
        'co2_ppm':        720.0,
        'temperature':    33.0,
        'humidity':       45.0,
        'vibration':      1,
        'ndvi':           0.12,
        'rainfall_mm':    0.0,
        'mining_activity': 8.5
    }

    result = full_predict(test_sensor)

    print("\n" + "="*50)
    print("  TERRA-SENSE PREDICTION RESULT")
    print("="*50)
    print(f"  Current CAR   : {result['car_current']}")
    print(f"  Risk Level    : {result['risk']}")
    print(f"  Alert         : {result['alert']}")
    print(f"  Anomaly       : {result['is_anomaly']}")
    print(f"  Recommendation: {result['recommendation']}")
    print("="*50)