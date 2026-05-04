# Renewable Energy Output Prediction Using ANN
Based on Negnevitsky - AI: A Guide to Intelligent Systems, Ch. 6

## Project Overview
This project builds an Artificial Neural Network (ANN) to predict hourly PV energy output (kWh) from weather and time-based input features. The workflow follows a supervised learning pipeline with chronological splitting, scaling, ANN training via backpropagation, and comparison against a linear regression baseline.

## Model Architecture
The implemented network follows the required structure:
- Input Layer: 5 features (`Air_Temp`, `Wind_Speed`, `Relative_Humidity`, `hour`, `month`)
- Hidden Layer 1: 10 neurons, ReLU activation
- Hidden Layer 2: 8 neurons, ReLU activation
- Dropout: 0.2 between hidden layers
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
- Year covered: 2020 to 2020

`maindata.csv` includes PV output power `P` (W), solar irradiance, temperature, and wind speed. `Humidity.csv` provides relative humidity for the same hourly period.

### Target Definition
- `P` is converted from watts to hourly energy output:
- `Energy_kWh = P / 1000.0`

## Preprocessing
- Parse timestamps and align both datasets to hourly resolution
- Inner join on `Timestamp`
- Convert numeric columns and handle missing values with interpolation + median fill
- Feature engineering: `hour`, `month`, `day_of_week`, `season`
- Chronological split: 70% train, 15% validation, 15% test
- MinMax scaling for features and target

## Evaluation
The ANN is evaluated against a linear regression baseline using:
- RMSE
- MAE
- R2 Score

Visual outputs include:
- Training and validation loss curves
- Predicted vs actual scatter plots (kWh)
- Feature-target correlation summary
- Model architecture diagram

## How to Run
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the `Datasets` folder contains:
- `maindata.csv`
- `Humidity.csv`

3. Open and run all cells in `renewable_energy_ann.ipynb`.

## Outputs
- `artifacts/ann_model.h5`: trained ANN model
- `artifacts/model_architecture.png`: architecture diagram
- Notebook tables and plots for metrics and comparisons

## Notes
- Model diagram export requires `pydot` and Graphviz support.
- For reproducible results, run the notebook from top to bottom in a fresh kernel.
