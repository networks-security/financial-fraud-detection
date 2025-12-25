import os
import sys
import joblib
import json
import pandas as pd

# Add paths for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks', 'saved_models'))

import transform_data_functions as utils_transform
import ml_training_functions as utils_training

if __name__ == "__main__":
    load_model = None

    if load_model == None:
        # Use absolute path based on script location
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        FOLDER_PATH = os.path.join(SCRIPT_DIR, "notebooks", "saved_models")
        FILE_PATH = os.path.join(FOLDER_PATH, "super_learner_model_iso_forest.pkl")

        if not os.path.isdir(FOLDER_PATH):
            raise NotADirectoryError(f"Error: The folder '{FOLDER_PATH}' does not exist.")
        if not os.path.isfile(FILE_PATH):
            raise FileNotFoundError(f"Error: The file '{FILE_PATH}' does not exist.")

        load_model = joblib.load(FILE_PATH)
        
    raw_transaction_record = json.loads(sys.argv[1]) # Retrieve new transaction input from banking system
    
    # Retrieve new transaction input from banking system
    try:
        # Create DataFrame and rename columns from JSON format to dataset format
        new_transaction_df = pd.DataFrame([raw_transaction_record])
        
        # Map JSON column names to dataset column names
        column_mapping = {
            'transactionId': 'TRANSACTION_ID',
            'txDatetime': 'TX_DATETIME',
            'customerId': 'CUSTOMER_ID',
            'terminalId': 'TERMINAL_ID',
            'txAmount': 'TX_AMOUNT',
            'txTimeSeconds': 'TX_TIME_SECONDS',
            'txTimeDays': 'TX_TIME_DAYS'
        }
        
        new_transaction_df = new_transaction_df.rename(columns=column_mapping)
        
        # Convert TX_DATETIME to datetime if it's a string
        if 'TX_DATETIME' in new_transaction_df.columns:
            new_transaction_df['TX_DATETIME'] = pd.to_datetime(new_transaction_df['TX_DATETIME'])
        
        # Add TX_FRAUD and TX_FRAUD_SCENARIO column (required for transform_new_df to work)
        if 'TX_FRAUD' not in new_transaction_df.columns:
            new_transaction_df['TX_FRAUD'] = 0
        if 'TX_FRAUD_SCENARIO' not in new_transaction_df.columns:
            new_transaction_df['TX_FRAUD_SCENARIO'] = 0
        
        # Transform and get prediction
        transformed_transaction_df = utils_transform.transform_new_df(new_transaction_df)
        
        input_features = load_model['input_features']
        # Ensure all required features are present, fill missing with 0
        for feature in input_features:
            if feature not in transformed_transaction_df.columns:
                transformed_transaction_df[feature] = 0
        
        transformed_transaction_df[input_features] = transformed_transaction_df[input_features].fillna(0)
        
        prediction = utils_training.get_super_learner_prediction(transformed_transaction_df, load_model)
        fraud_prob = float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction) # Convert prediction to float if needed
        
        # Business Rule:
        # - If prob >= 0.5 -> Mark as fraud.
        # - If prob >= 0.2 -> Mark as suspicious, send alert.
        # - If prob < 0.2 -> Mark as legitimate.
        
        if fraud_prob >= 0.5:
            status = "fraud"
        elif fraud_prob >= 0.2:
            status = "suspicious"
        else:
            status = "legitimate"
        
        result = {
            "status": "success",
            "fraud_probability": float(fraud_prob),
            "classification": status
        }
        print(json.dumps(result))
    
    except Exception as e:
        error_response = {
            "status": "error",
            "message": f"Prediction failed: {str(e)}",
            "fraud_probability": None
        }
        print(json.dumps(error_response))
        sys.exit(1)