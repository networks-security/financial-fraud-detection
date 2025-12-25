import os
import datetime
import pandas as pd
import pickle

def save_transformed_data_to_file(DIR_OUTPUT, start_date, transactions_df):
    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)

    start_date = datetime.datetime.strptime("2018-04-01", "%Y-%m-%d")

    for day in range(transactions_df.TX_TIME_DAYS.max()+1):
        
        transactions_day = transactions_df[transactions_df.TX_TIME_DAYS==day].sort_values('TX_TIME_SECONDS')
        
        date = start_date + datetime.timedelta(days=day)
        filename_output = date.strftime("%Y-%m-%d")+'.pkl'
        
        # Protocol=4 required for Google Colab
        transactions_day.to_pickle(DIR_OUTPUT+filename_output, protocol=4)

def read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE):
    
    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if f>=BEGIN_DATE+'.pkl' and f<=END_DATE+'.pkl']

    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)
    
    df_final=df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True,inplace=True)
    #  Note: -1 are missing values for real world data 
    df_final=df_final.replace([-1],0)
    
    return df_final

def save_model_data(file_path, model_data):
    filehandler = open(file_path, 'wb') 
    pickle.dump(model_data, filehandler)
    filehandler.close()

def load_model_data(file_path):
    
    try:
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)
        
        # Handle both single object and tuple of 2 objects
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            df_dictionary, execution_times = loaded_data
        else:
            # Single object case (e.g., just the model dictionary)
            df_dictionary = loaded_data
            execution_times = None
        
        print("Object successfully loaded from pickle file:")
        return df_dictionary, execution_times
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return None, None