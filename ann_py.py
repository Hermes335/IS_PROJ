# %% [markdown]
# # Renewable Energy Output Prediction Using Artificial Neural Networks
# 
# Based on Negnevitsky — AI: A Guide to Intelligent Systems, Ch. 6
# 
# ## Problem Statement
# Accurate hourly energy-output forecasting helps improve solar planning, grid operations, and system sizing. This project trains an ANN to predict PV energy output (kWh) from weather and time-based inputs.
# 
# ## Objectives
# - Build a multilayer perceptron (MLP) with 2 hidden layers using TensorFlow/Keras
# - Train the network using backpropagation on real-world hourly PV and weather data
# - Predict PV energy output (kWh) from weather input features
# - Compare ANN performance against a linear regression baseline
# - Evaluate using RMSE, MAE, and R² metrics

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import joblib

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

OUTPUT_DIR = Path('artifacts')

# Weather and time-based feature columns
# Add or Remove Irradiance to check changes
FEATURE_COLUMNS = ['Air_Temp', 'Wind_Speed', 'Relative_Humidity', 'hour', 'month', 'is_daylight', 'irradiance']
# Target column to predict (energy output in kWh)
TARGET_COLUMN = 'Energy_kWh'

EPOCHS = 150
BATCH_SIZE = 32
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
LEARNING_RATE = 0.001
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)




# %% [markdown]
# ## 1. Load and preprocess data
# 
# The dataset is sorted by time, missing values are handled, and time features such as hour and month are extracted from the datetime column.

# %%
# The datasets are already provided in the Datasets folder.
dataset_folder = 'Datasets'
print(f"Dataset folder: {dataset_folder}")
import os
for file in os.listdir(dataset_folder):
    print(f"  - {file}")


# %%
# Read PV output dataset and humidity dataset.
# maindata.csv has metadata rows before the actual header.
df_main = pd.read_csv("Datasets/maindata.csv", skiprows=10)
print("Main dataset columns:", df_main.columns.tolist())

df_hum = pd.read_csv("Datasets/Humidity.csv", skiprows=2)
print("Humidity columns:", df_hum.columns.tolist())

# Parse PVGIS timestamp format: YYYYMMDD:HHMM
# PVGIS is in UTC. Shift +8 hours to align with the local Philippines time of the Humidity data.
df_main['Timestamp'] = pd.to_datetime(df_main['time'], format='%Y%m%d:%H%M', errors='coerce')
df_main = df_main.dropna(subset=['Timestamp'])
df_main['Timestamp'] = df_main['Timestamp'].dt.floor('h') + pd.Timedelta(hours=8)

# Parse humidity timestamp
df_hum['Timestamp'] = pd.to_datetime(df_hum[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df_hum['Timestamp'] = df_hum['Timestamp'].dt.floor('h')

print("\nMain dataset first rows:")
print(df_main.head(2))

print("\nHumidity first rows:")
print(df_hum.head(2))

# %%
# Merge on the aligned hourly timestamp
df = pd.merge(df_main, df_hum, on='Timestamp', how='inner')

# Rename raw columns to modeling names
df = df.rename(columns={
    'T2m': 'Air_Temp',
    'WS10m': 'Wind_Speed',
    'Relative Humidity': 'Relative_Humidity',
    'P': 'PV_Power_W',
    'G(i)': 'irradiance',
    'H_sun': 'sun_elevation'
})

# Convert PV power (W) to hourly energy output (kWh)
df['PV_Power_W'] = pd.to_numeric(df['PV_Power_W'], errors='coerce')
df['Energy_kWh'] = df['PV_Power_W'] / 1000.0

df['irradiance'] = pd.to_numeric(df.get('irradiance', 0), errors='coerce')
df['sun_elevation'] = pd.to_numeric(df.get('sun_elevation', 0), errors='coerce')
df['is_daylight'] = (df['irradiance'] > 10.0).astype(int)

TARGET_COLUMN = 'Energy_kWh'

df = df.sort_values('Timestamp').reset_index(drop=True)
df = df.dropna(subset=['Timestamp', TARGET_COLUMN])

df['hour'] = df['Timestamp'].dt.hour
df['month'] = df['Timestamp'].dt.month
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['season'] = (df['month'] % 12 // 3 + 1)

numeric_columns = ['Energy_kWh', 'Air_Temp', 'Wind_Speed', 'Relative_Humidity', 'irradiance', 'sun_elevation']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df[numeric_columns] = df[numeric_columns].interpolate(method='linear', limit_direction='both')
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median(numeric_only=True))

df.head()


# %%
print("=" * 70)
print("DATASET OVERVIEW & STATISTICS")
print("=" * 70)
print(f"\nDataset Shape: {df.shape[0]:,} samples × {df.shape[1]} variables")
print(f"Date Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"Duration: {(df['Timestamp'].max() - df['Timestamp'].min()).days} days")

print("\n" + "-" * 70)
print("Feature Statistics:")
print("-" * 70)
stats_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T
print(stats_df[['mean', 'std', 'min', 'max']].to_string())

print("\n" + "-" * 70)
print("Feature Correlation with Target (Energy kWh):")
print("-" * 70)
correlations = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(key=abs, ascending=False)
for feat, corr in correlations.items():
    print(f"  {feat:20s}: {corr:7.4f}")
print("=" * 70)

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

corr_matrix = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=axes[0], cbar=False)
axes[0].set_title('Feature-Target Correlation')

monthly_profile = df.groupby('month')[TARGET_COLUMN].mean()
sns.lineplot(x=monthly_profile.index, y=monthly_profile.values, marker='o', ax=axes[1])
axes[1].set_title('Average Energy Output by Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Energy_kWh')

hourly_profile = df.groupby('hour')[TARGET_COLUMN].mean()
sns.lineplot(x=hourly_profile.index, y=hourly_profile.values, marker='o', ax=axes[2])
axes[2].set_title('Average Energy Output by Hour')
axes[2].set_xlabel('Hour of Day')
axes[2].set_ylabel('Energy_kWh')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1b. Presentation Visuals
# 
# These plots make the data story easier to explain in a presentation: how the inputs relate to the target, and how PV energy changes across time.

# %% [markdown]
# ## 2. Chronological split and scaling
# 
# The split is time-based, not random, so future observations are never used in training. Inputs and target values are then scaled with MinMaxScaler.

# %%
def chronological_split(frame, validation_ratio=0.15, test_ratio=0.15):
    if validation_ratio + test_ratio >= 1:
        raise ValueError('Validation and test ratios must sum to less than 1.')

    total_rows = len(frame)
    test_size = int(round(total_rows * test_ratio))
    validation_size = int(round(total_rows * validation_ratio))
    train_end = total_rows - validation_size - test_size

    if train_end <= 0:
        raise ValueError('Not enough rows to create train/validation/test splits.')

    train_df = frame.iloc[:train_end].copy()
    validation_df = frame.iloc[train_end:train_end + validation_size].copy()
    test_df = frame.iloc[train_end + validation_size:].copy()
    return train_df, validation_df, test_df

train_df, validation_df, test_df = chronological_split(df, VALIDATION_RATIO, TEST_RATIO)

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train = feature_scaler.fit_transform(train_df[FEATURE_COLUMNS])
X_val = feature_scaler.transform(validation_df[FEATURE_COLUMNS])
X_test = feature_scaler.transform(test_df[FEATURE_COLUMNS])

y_train = target_scaler.fit_transform(train_df[[TARGET_COLUMN]])
y_val = target_scaler.transform(validation_df[[TARGET_COLUMN]])
y_test = target_scaler.transform(test_df[[TARGET_COLUMN]])

X_train.shape, X_val.shape, X_test.shape

# %%
# Validate required columns exist
required_cols = ['irradiance', 'sun_elevation']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

df['irradiance'] = pd.to_numeric(df['irradiance'], errors='coerce')
df['sun_elevation'] = pd.to_numeric(df['sun_elevation'], errors='coerce')

# %% [markdown]
# ## 3. Build the ANN model
# 
# The model follows the requested architecture: 10 hidden units, then 8 hidden units, then a linear output neuron for regression.

# %%
def build_ann_model(input_dim, learning_rate=0.001):
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.2), 
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='linear')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

ann_model = build_ann_model(len(FEATURE_COLUMNS), learning_rate=LEARNING_RATE)
ann_model.summary()

# %%
print("\n" + "=" * 70)
print("MODEL ARCHITECTURE VISUALIZATION")
print("=" * 70)

output_path = OUTPUT_DIR / 'model_architecture.png'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    keras.utils.plot_model(
        ann_model,
        to_file=str(output_path),
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        dpi=120,
    )
    if output_path.exists():
        from IPython.display import Image, display
        display(Image(filename=str(output_path)))
    else:
        print(f"Model diagram could not be found at {output_path}.")
except Exception as exc:
    print(f"Model diagram export skipped: {exc}")


# %% [markdown]
# ## 4. Train the ANN
# 
# EarlyStopping restores the best weights from validation loss, which helps reduce overfitting.

# %%
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

history = ann_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# %%
print("\n" + "=" * 70)
print("DATA SHAPES FOR TRAINING")
print("=" * 70)
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}")
print(f"X_test shape:  {X_test.shape}   | y_test shape:  {y_test.shape}")
print(f"\nTraining Configuration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Early Stopping Patience: 10")
print("=" * 70 + "\n")

# %% [markdown]
# ## 5. Evaluate the ANN and compare it with linear regression
# 
# Metrics are computed on the inverse-transformed test predictions so they are reported in the original kWh scale.

# %%
def inverse_target(values):
    return target_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(-1)

def evaluate_predictions(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }

y_test_true = inverse_target(y_test)
ann_pred = inverse_target(ann_model.predict(X_test))
ann_metrics = evaluate_predictions(y_test_true, ann_pred)

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train.ravel())
baseline_pred = inverse_target(baseline_model.predict(X_test))
baseline_metrics = evaluate_predictions(y_test_true, baseline_pred)

print('ANN metrics:', ann_metrics)
print('Baseline metrics:', baseline_metrics)

# %%
print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

results_table = pd.DataFrame({
    'Model': ['ANN', 'Linear Regression'],
    'RMSE': [f"{ann_metrics['rmse']:.4f}", f"{baseline_metrics['rmse']:.4f}"],
    'MAE': [f"{ann_metrics['mae']:.4f}", f"{baseline_metrics['mae']:.4f}"],
    'R² Score': [f"{ann_metrics['r2']:.4f}", f"{baseline_metrics['r2']:.4f}"]
})
print(results_table.to_string(index=False))

improvement_rmse = ((baseline_metrics['rmse'] - ann_metrics['rmse']) / baseline_metrics['rmse']) * 100
improvement_r2 = ((ann_metrics['r2'] - baseline_metrics['r2']) / baseline_metrics['r2']) * 100 if baseline_metrics['r2'] > 0 else 0

print(f"\nANN Improvement over Baseline:")
print(f"  RMSE:   {improvement_rmse:+.2f}% {'better' if improvement_rmse > 0 else 'worse'}")
print(f"  R² Score: {improvement_r2:+.2f}% {'better' if improvement_r2 > 0 else 'worse'}")
print("=" * 70)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (Correlation with Target)")
print("=" * 70)
feature_corr = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Correlation': [df[feat].corr(df[TARGET_COLUMN]) for feat in FEATURE_COLUMNS]
}).sort_values('Correlation', key=abs, ascending=False)

for idx, row in feature_corr.iterrows():
    bar = '█' * int(abs(row['Correlation']) * 50)
    print(f"{row['Feature']:20s}: {row['Correlation']:7.4f}  {bar}")
print("=" * 70)

# %%
residuals = y_test_true - ann_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(x=ann_pred, y=residuals, alpha=0.5, ax=axes[0])
axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0].set_title('Residuals vs Predicted')
axes[0].set_xlabel('Predicted Energy Output (kWh)')
axes[0].set_ylabel('Residual (Actual - Predicted)')

sns.histplot(residuals, bins=30, kde=True, ax=axes[1], color='steelblue')
axes[1].set_title('Residual Distribution')
axes[1].set_xlabel('Residual (kWh)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Visualize results and save the trained model
# 
# The plots below show training behavior and predicted-vs-actual performance. The trained ANN is also saved to disk as an `.h5` file.

# %%
def plot_training_history(history_obj):
    plt.figure(figsize=(10, 6))
    plt.plot(history_obj.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history_obj.history:
        plt.plot(history_obj.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('ANN Training Progress: Loss Convergence', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_predicted_vs_actual(y_true, y_pred, title):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, s=30, alpha=0.6)
    min_value = min(np.min(y_true), np.min(y_pred))
    max_value = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Energy Output (kWh)', fontsize=12)
    plt.ylabel('Predicted Energy Output (kWh)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\n" + "=" * 70)
print("VISUALIZATION: TRAINING HISTORY & PREDICTIONS")
print("=" * 70 + "\n")

plot_training_history(history)
plot_predicted_vs_actual(y_test_true, ann_pred, 'ANN Model: Predicted vs Actual Energy Output (kWh)')
plot_predicted_vs_actual(y_test_true, baseline_pred, 'Linear Regression Baseline: Predicted vs Actual Energy Output (kWh)')

model_path = OUTPUT_DIR / 'ann_model.h5'
ann_model.save(model_path)
print(f'Saved model to {model_path}')

# %% [markdown]
# ## 7. Summary

# %%
import math

def _inv(arr):
    try:
        return target_scaler.inverse_transform(np.asarray(arr).reshape(-1, 1)).reshape(-1)
    except Exception:
        return None

def _compute_metrics(model):
    if model is None or 'X_test' not in globals() or 'y_test' not in globals():
        return {'rmse': math.nan, 'mae': math.nan, 'r2': math.nan}
    try:
        y_true = _inv(y_test)
        y_pred = _inv(model.predict(X_test))
        return evaluate_predictions(y_true, y_pred) if (y_true is not None and y_pred is not None) else {'rmse': math.nan, 'mae': math.nan, 'r2': math.nan}
    except Exception:
        return {'rmse': math.nan, 'mae': math.nan, 'r2': math.nan}

# Always recompute to avoid stale metrics from previous target definitions
ann_metrics = _compute_metrics(globals().get('ann_model'))
baseline_metrics = _compute_metrics(globals().get('baseline_model'))

ann_rmse = ann_metrics.get('rmse', math.nan)
ann_mae = ann_metrics.get('mae', math.nan)
ann_r2 = ann_metrics.get('r2', math.nan)
baseline_rmse = baseline_metrics.get('rmse', math.nan)

improvement_rmse = ((baseline_rmse - ann_rmse) / baseline_rmse * 100) if (baseline_rmse and not math.isnan(baseline_rmse) and baseline_rmse > 0) else 0.0

try:
    feature_corr = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Correlation': [df[f].corr(df[TARGET_COLUMN]) for f in FEATURE_COLUMNS]
    }).sort_values('Correlation', key=abs, ascending=False)
    top_feat = f"{feature_corr.iloc[0]['Feature']} (r={feature_corr.iloc[0]['Correlation']:.4f})"
except Exception:
    top_feat = 'n/a'

lines = [
    "=" * 70,
    "PROJECT SUMMARY & CONCLUSIONS",
    "=" * 70,
    f"Goal achieved: Predicting {TARGET_COLUMN} from weather/time features",
    f"Samples: {df.shape[0] if 'df' in globals() else 'n/a'}",
    f"Features: {', '.join(FEATURE_COLUMNS)}",
    "",
    "Architecture: 10 -> 8 -> 1, dropout 20%, Adam, MSE",
    "",
    f"ANN RMSE: {ann_rmse:.4f} kWh",
    f"ANN MAE:  {ann_mae:.4f} kWh",
    f"ANN R2:   {ann_r2:.4f}",
    f"Baseline RMSE: {baseline_rmse:.4f} kWh",
    f"RMSE improvement vs baseline: {improvement_rmse:.2f}%",
    "",
    f"Top correlated feature: {top_feat}",
    "=" * 70,
]
print("\n".join(lines))



