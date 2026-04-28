# ── Renewable Energy ANN — Starter Code ──────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras

# ── Step 1: Load & Preprocess Data ──────────────────────────
df = pd.read_csv('solar_data.csv', parse_dates=['datetime'])
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
features = ['irradiance','temperature','wind_speed','humidity','hour']
X = df[features].values
y = df['energy_kwh'].values.reshape(-1, 1)

scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ── Step 2: Build ANN Model ──────────────────────────────────
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    keras.layers.Dense(8,  activation='relu'),
    keras.layers.Dense(1,  activation='linear')  # regression output
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ── Step 3: Train ────────────────────────────────────────────
history = model.fit(X_train, y_train, epochs=150,
    batch_size=32, validation_data=(X_val, y_val), verbose=1)

# ── Step 4: Evaluate ─────────────────────────────────────────
y_pred = scaler_y.inverse_transform(model.predict(X_test))
y_true = scaler_y.inverse_transform(y_test)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f'RMSE: {rmse:.3f} | R²: {r2_score(y_true,y_pred):.3f}')
