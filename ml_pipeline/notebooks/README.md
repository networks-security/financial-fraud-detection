
# Machine Learning Artifacts

This directory contains all models, scalers, shared functions, and simulated data used in the deep learning fraud detection project.

## Contents:
- `saved_models/`: Contains saved PyTorch models (SimpleAutoencoder, SimpleFraudMLPWithDropout) and scikit-learn models (IsolationForest).
- `saved_scalers/`: Contains fitted `StandardScaler` objects for feature preprocessing.
- `shared_functions.py`: Python script containing helper functions used across the project.
- `data/simulated-data-raw/`: Raw simulated transaction data.
- `data/simulated-data-transformed/`: Transformed simulated transaction data with engineered features.
