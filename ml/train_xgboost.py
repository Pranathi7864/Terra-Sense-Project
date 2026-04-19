import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os

df = pd.read_csv('data/singrauli_sensor_data.csv')
print(f"Dataset loaded: {len(df):,} rows")
features = ['soil_moisture', 'co2_ppm', 'temperature','humidity', 'vibration', 'ndvi','rainfall_mm', 'mining_activity']
X = df[features].fillna(0)
y = df['CAR']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, verbose=False)
preds = model.predict(X_test)
r2   = r2_score(y_test, preds)
print(f"R² Score : {r2:.4f}")


os.makedirs('ml', exist_ok=True)
pickle.dump(model, open('ml/car_model.pkl', 'wb'))
print("Trained....")