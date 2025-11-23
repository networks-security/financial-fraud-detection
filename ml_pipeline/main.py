import os
import sys
import joblib
import json
import pandas as pd

sys.path.append('./notebooks')
sys.path.append('./notebooks/models')
sys.path.append('./notebooks/utils')

import transform_data_functions as utils_transform
import ml_training_functions as utils_training

if __name__ == "__main__":
    load_model = None

    if load_model == None:
        FOLDER_PATH = "./notebooks/models"
        FILE_PATH = FOLDER_PATH + "/super_learner_model.pkl"

        if not os.path.isdir(FOLDER_PATH):
            raise NotADirectoryError(f"Error: The folder '{FOLDER_PATH}' does not exist.")
        if not os.path.isfile(FILE_PATH):
            raise FileNotFoundError(f"Error: The file '{FILE_PATH}' does not exist.")

        load_model = joblib.load(FILE_PATH)

    raw_transaction_record = json.loads(sys.argv[1]) # Retrieve new transaction input from bankiung system
    new_transaction_df = pd.DataFrame(raw_transaction_record)
    transformed_transaction_df = utils_transform.transform_new_df(new_transaction_df)
    prediction = utils_training.get_super_learner_prediction(transformed_transaction_df, load_model)

    '''
    - prediction will return as probability of fraud for that particular trasaction.
        
    - Business Rule should be:
        - If prob >= 0.5 -> Mark as fraud.
        - If prob >= 0.2 -> Mark as sus, send alert.
    '''

    print(prediction)