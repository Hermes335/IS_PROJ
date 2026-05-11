# Renewable Energy Output Prediction Using ANN
Based on Negnevitsky - AI: A Guide to Intelligent Systems, Ch. 6

## Project Overview
This project builds an Artificial Neural Network (ANN) to predict hourly PV energy output (kWh) from weather and time-based input features. The workflow follows a supervised learning pipeline with chronological splitting, scaling, ANN training via backpropagation, and comparison against a linear regression baseline.

## Model Architecture
The implemented network follows the required structure:
- Input Layer: 7 features (`Air_Temp`, `Wind_Speed`, `Relative_Humidity`, `hour`, `month`, `is_daylight`, `irradiance`)
- Hidden Layer 1: 10 neurons, ReLU activation
- Dropout: 0.2
- Hidden Layer 2: 8 neurons, ReLU activation
- Dropout: 0.2
- Output Layer: 1 neuron, linear activation (predicting `Energy_kWh`)
- Optimizer: Adam (`learning_rate=0.001`)
- Loss: Mean Squared Error (MSE)
- Early Stopping: patience of 10 epochs, restore best weights

## Datasets
The notebook merges two hourly datasets collected for the same site and year:
- `Datasets/maindata.csv` from the JRC PVGIS tools: [https://re.jrc.ec.europa.eu/pvg_tools/en/#HR](https://re.jrc.ec.europa.eu/pvg_tools/en/#HR)
- `Datasets/Humidity.csv` from the NREL NSRDB data viewer: [https://nsrdb.nrel.gov/data-viewer](https://nsrdb.nrel.gov/data-viewer)

Site information:
- Location: Iloilo City
- Latitude: 10.743
- Longitude: 122.529
- Year covered: 2020

`maindata.csv` includes PV output power `P` (W), solar irradiance, temperature, and wind speed. `Humidity.csv` provides relative humidity for the same hourly period.

### Target Definition
- `P` is converted from watts to hourly energy output:
- `Energy_kWh = P / 1000.0`

### Important Note on Irradiance
The `irradiance` feature has a correlation of ~0.999 with the target, making it an almost-deterministic predictor. This causes linear regression to outperform the ANN numerically. For realistic deployment scenarios (where direct irradiance may be unavailable), see `report.md` for comparison of model performance with and without irradiance.

## Preprocessing
- Parse timestamps and align both datasets to hourly resolution
- Timezone fix: PVGIS timestamps (UTC) shifted +8 hours to match local Philippine time
- Inner join on `Timestamp`
- Convert numeric columns and handle missing values with interpolation + median fill
- Feature engineering: `hour`, `month`, `day_of_week`, `season`, `is_daylight`
- Chronological split: 70% train, 15% validation, 15% test (prevents data leakage)
- MinMax scaling for features and target (fit on train, transform on val/test)

## Evaluation
The ANN is evaluated against a linear regression baseline using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

Visual outputs include:
- Training and validation loss curves
- Predicted vs actual scatter plots (kWh)
- Feature-target correlation heatmap
- Residual distribution plots
- Model architecture diagram

## How to Run
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the `Datasets` folder contains:
- `maindata.csv`
- `Humidity.csv`

3. Run either:
   - **Option A:** Open and run all cells in `renewable_energy_ann.ipynb`
   - **Option B:** Run `python ann_py.py` in terminal

## Outputs
- `artifacts/ann_model.h5`: trained ANN model
- `artifacts/feature_scaler.joblib`: scaler for input features (required for inference)
- `artifacts/target_scaler.joblib`: scaler for target variable (required for inference)
- `artifacts/model_architecture.png`: architecture diagram
- Notebook tables and plots for metrics and comparisons

## Reproducibility
- Set `SEED = 42` at the top of the script/notebook
- Run from top to bottom in a fresh kernel for consistent results

## Files
| File | Description |
|------|-------------|
| `renewable_energy_ann.ipynb` | Jupyter notebook with full pipeline |
| `ann_py.py` | Standalone Python script (same content) |
| `report.md` | Comparison of model performance with/without irradiance |
| `requirements.txt` | Python dependencies |

## Notes
- Model diagram export requires `pydot` and Graphviz support.
- For deployment, load both the model AND scalers (both are required for inference).
- Before production use, validate across multiple years and locations.