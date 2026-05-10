# PV Energy ANN — Comparison Report

This short report compares two modeling scenarios trained in `renewable_energy_ann.ipynb`:

- With `irradiance` included as a feature (PVGIS irradiance).
- Without `irradiance` (deployment-realistic feature set).

---

## Key metrics

Metrics computed on the held-out chronological test set (inverse-transformed to kWh):

- With irradiance included
  - ANN: RMSE = 0.0588 kWh, MAE = 0.0425 kWh, R² = 0.9484
  - Linear regression (baseline): RMSE = 0.0073 kWh, MAE = 0.0058 kWh, R² = 0.9992

- Without irradiance (no-irradiance / realistic)
  - ANN: RMSE = 0.1213 kWh, MAE ≈ (see notebook), R² = 0.7801
  - Linear regression (baseline): RMSE = 0.2035 kWh, R² = 0.3807


## Interpretation

- Including `irradiance` dramatically improves all metrics for both ANN and the linear baseline. The baseline becomes nearly perfect (R² ≈ 0.999) because the PVGIS `irradiance` feature is an almost-deterministic predictor of PV energy in the dataset (high feature-target correlation).

- Removing `irradiance` is the fairer, deployment-realistic test: the ANN still substantially outperforms the linear baseline (lower RMSE and much higher R²), demonstrating the ANN can learn nonlinear relationships from weather/time features.

- In short: absolute error is worse when `irradiance` is removed (as expected), but the ANN gains meaningful generalization advantage over a linear baseline in that no-irradiance scenario.
