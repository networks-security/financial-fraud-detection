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

def save_model_to_file(performances_df_dictionary, execution_times):
    filehandler = open('performances_model_selection.pkl', 'wb') 
    pickle.dump((performances_df_dictionary, execution_times), filehandler)
    filehandler.close()

def load_model_to_file():
    file_path = 'performances_model_selection.pkl'
    try:
        with open(file_path, 'rb') as file:
            performances_df_dictionary, execution_times = pickle.load(file)
        print("Object successfully loaded from pickle file:")
        return performances_df_dictionary, execution_times
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return None, None

def save_model_performance_result_to_file(performances_df_dictionary, execution_times):
    filehandler = open('performances_ensembles.pkl', 'wb') 
    pickle.dump((performances_df_dictionary, execution_times), filehandler)
    filehandler.close()

def load_model_performance_summary():
    file_path = 'performances_ensembles.pkl'
    try:
        with open(file_path, 'rb') as file:
            performances_df_dictionary, execution_times = pickle.load(file)
        print("Object successfully loaded from pickle file:")
        return performances_df_dictionary, execution_times
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return None, None