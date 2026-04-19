import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import pickle
import os

print("Training Isolation Forest Anomaly Detector...")

df = pd.read_csv('data/singrauli_sensor_data.csv')

features = [
    'soil_moisture', 'co2_ppm', 'temperature',
    'humidity', 'vibration', 'ndvi',
    'rainfall_mm', 'mining_activity'
]

X = df[features].fillna(0)

# ── Train Isolation Forest ────────────────────
# contamination = expected % of anomalies in data
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # 5% of data expected as anomalies
    random_state=42,
    n_jobs=-1
)
model.fit(X)

# ── Predict anomalies ─────────────────────────
# Returns: -1 = anomaly, 1 = normal
preds = model.predict(X)
anomaly_count  = (preds == -1).sum()
normal_count   = (preds ==  1).sum()

print(f"\nResults:")
print(f"Normal readings  : {normal_count:,}")
print(f"Anomaly readings : {anomaly_count:,}")
print(f"Anomaly rate     : {anomaly_count/len(preds)*100:.2f}%")

# ── Check anomalies vs Critical zones ─────────
df['anomaly'] = preds
anomalies = df[df['anomaly'] == -1]
print(f"\nAnomalies per zone:")
print(anomalies['zone'].value_counts().to_string())

print(f"\nAnomaly CAR stats:")
print(f"Mean CAR in anomalies : {anomalies['CAR'].mean():.3f}")
print(f"Min CAR in anomalies  : {anomalies['CAR'].min():.3f}")

# ── Save ──────────────────────────────────────
os.makedirs('ml', exist_ok=True)
pickle.dump(model, open('ml/isolation_model.pkl', 'wb'))
print("Trained.....")