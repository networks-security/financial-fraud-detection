import sys
import os
from typing import Dict, Any
import joblib
import pandas as pd
from ml_service.ml_pipeline.notebooks.utils.transform_data_functions import (
    transform_new_df,
)
from ml_service.ml_pipeline.notebooks.utils.ml_training_functions import (
    get_super_learner_prediction,
)
from ml_service.core.schemas.transaction_model import TransactionModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def detect_fraud(transaction: TransactionModel) -> Dict[str, Any]:
    """
    Runs the full fraud detection pipeline on a single transaction.

    Args:
        transaction (TransactionModel): Input transaction following the defined schema.

    Returns:
        Dict containing status, fraud_probability, and classification.
    """
    # Resolve model path relative to this file's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        script_dir, "notebooks", "saved_models", "super_learner_model_iso_forest.pkl"
    )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load the trained super learner model (assumed to be a dict with 'model' and 'input_features')
    model = joblib.load(model_path)

    try:

        # Expected column mapping from API/schema to training dataset
        column_mapping = {
            "transaction_id": "TRANSACTION_ID",
            "tx_datetime": "TX_DATETIME",
            "customer_id": "CUSTOMER_ID",
            "terminal_id": "TERMINAL_ID",
            "tx_amount": "TX_AMOUNT",
            "tx_time_seconds": "TX_TIME_SECONDS",
            "tx_time_days": "TX_TIME_DAYS",
        }

        df = pd.DataFrame([transaction])
        df = df.rename(columns=column_mapping)

        # Ensure datetime format
        if "TX_DATETIME" in df.columns:
            df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

        # Add required placeholder columns for transformation function compatibility
        df["TX_FRAUD"] = 0
        df["TX_FRAUD_SCENARIO"] = 0

        # Import transformation utilities (adjust paths if your structure differs)

        sys.path.insert(0, os.path.join(script_dir, "notebooks"))
        sys.path.insert(0, os.path.join(script_dir, "notebooks", "utils"))

        # Apply the same transformations used during training
        transformed_df = transform_new_df(df)

        # Ensure all expected input features are present
        input_features = model["input_features"]
        for feature in input_features:
            if feature not in transformed_df.columns:
                transformed_df[feature] = 0

        transformed_df[input_features] = transformed_df[input_features].fillna(0)

        # Get prediction from super learner
        prediction = get_super_learner_prediction(transformed_df, model)
        fraud_prob = (
            float(prediction[0])
            if hasattr(prediction, "__iter__")
            else float(prediction)
        )

        # Apply business rules
        if fraud_prob >= 0.5:
            classification = "fraud"
        elif fraud_prob >= 0.2:
            classification = "suspicious"
        else:
            classification = "legitimate"

        return {
            "status": "success",
            "fraud_probability": float(fraud_prob),
            "classification": classification,
        }

    # pylint: disable=broad-except
    except Exception as e:
        return {
            "status": "error",
            "fraud_probability": None,
            "classification": None,
            "message": f"An unexpected error occurred: {str(e)}",
        }
