# SCTRiX
Source code for SCTRiX, machine-learning prediction of Seychelles-Chagos Thermocline Ridge (SCTR) D20 anomalies.

## Overview

SCTRiX is an XGBoost-based machine learning framework developed to predict monthly anomalies of the 20°C isotherm depth (D20) over the Seychelles–Chagos Thermocline Ridge (SCTR).

The framework uses physically guided predictors from the SCTR region, Indian Ocean, and Pacific Ocean, including local thermocline variability, oceanic and atmospheric climate signals, and large-scale climate indices.

## Workflow

The SCTRiX framework consists of two main steps:

### 1. Feature engineering

The feature engineering scripts:

- load ORAS5 ocean reanalysis and ERA5 atmospheric reanalysis datasets
- calculate monthly anomalies
- extract EOF-based predictors
- construct lead-time dependent D20 prediction targets
- generate training and testing datasets

Location: feature_engineering/


### 2. Machine learning model

The model scripts:

- train XGBoost regression models
- optimize hyperparameters using RandomizedSearchCV
- evaluate prediction skill for lead times of 1–12 months
- calculate feature importance and SHAP values
- perform grouped leave-one-feature-out (LOFO) experiments

Location: model/

## Repository structure

SCTRiX/
├── feature_engineering/
│   ├── Feature_engineering_loopseed.py
│   └── Feature_engineering_test.py
│
├── model/
│   ├── Model_loopseed.py
│   ├── Model_lofo_loopseed.py
│   └── Model_test.py
│
└── README.md


## Data availability

The input datasets are not included in this repository due to their size and distribution policies.

Required datasets:

- ORAS5 ocean reanalysis
- ERA5 atmospheric reanalysis

After downloading the datasets, place them under:
data/
└── input/

The generated feature files should be placed under:
data/
└── feature/


## Requirements

Python packages required for running the scripts include:

- numpy
- pandas
- xarray
- netCDF4
- scikit-learn
- xgboost
- eofs
- scipy
- matplotlib
- cartopy
- joblib

## Citation

If you use this code, please cite:

[Manuscript information will be added after publication]

## License

This code is provided for research and educational purposes.

