import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from sklearn.preprocessing import StandardScaler

import xgboost as xgb

import datetime # Ensure datetime is imported for date operations
import time # Ensure time is imported for time operations
import random # Ensure random is imported for seeding

import warnings
warnings.filterwarnings("ignore")

# Define DEVICE globally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Helper functions
# 
# This chapter defines a number of helper functions that are used throughout the book.
# You can refer to this chapter if you are unsure of the definition or parameters of the functions.
# 
# Let us first import some common libraries


# ## Plotting functions

# ### Plot transactions on a map
# The following function displays the coordinates of the fraudulent transactions, and their associated terminal locations.

def plot_transactions_on_map(transactions_df, 
                             terminals_df, 
                             column_to_use='TX_FRAUD', 
                             start_date = "2018-04-01", 
                             end_date = "2018-04-01"):

    # Filter transactions to plot only fraudulent transactions and to a specific date range
    
    transactions_to_plot_df=transactions_df.loc[(transactions_df.TX_DATETIME >= start_date) & 
                                            (transactions_df.TX_DATETIME <= end_date) & 
                                            (transactions_df[column_to_use]==1)]
    
    # Plot terminals
    
    if len(terminals_df.loc[terminals_df.TERMINAL_ID.isin(transactions_to_plot_df.TERMINAL_ID.unique())])>0:
        ax = terminals_df.loc[terminals_df.TERMINAL_ID.isin(transactions_to_plot_df.TERMINAL_ID.unique())].plot(x='x_terminal_id', y='y_terminal_id', kind='scatter', c='red', label='Fraudulent terminals', figsize=(10,10))
    else:
        ax = terminals_df.plot(x='x_terminal_id', y='y_terminal_id', kind='scatter', c='black', label='All terminals', figsize=(10,10))


    # Plot transactions
    
    transactions_to_plot_df.plot(x='x_terminal_id', y='y_terminal_id', kind='scatter', c='orange', label='Fraudulent transactions', ax=ax)
    
    plt.title('Fraudulent transactions and terminals on a map')
    plt.xlabel('x_location')
    plt.ylabel('y_location')
    plt.show()


# ### Plot losses of autoencoder model

def plot_losses(train_losses, valid_losses):
    
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend()
    plt.title('Losses')
    plt.show()


# ## Performance functions

# ### Plot Precision-Recall curve
# The following function plots the Precision-Recall curve, with the average precision as title

def plot_pr_curve(y_true, y_pred, figsize=(10, 10)):
    
    if len(y_true.shape)>1:
        y_true=y_true.squeeze()

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)

    plt.figure(figsize=figsize)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


# ### Card Precision Top K
# The Card Precision Top K metric measures the proportion of fraudulent cards that are identified by the model in the top K highest scored transactions.
# 
# If a fraudulent card appears multiple times in the top K transactions, it is only counted once towards the numerator.
# 
# If a fraudulent card does not appear in the top K transactions, it is not counted at all.

def card_precision_top_k(y_true, y_pred, top_k):
    
    # Sort the predictions by score in descending order
    sorted_idx = np.argsort(y_pred)[::-1]
    
    # Get the top K transactions
    top_k_idx = sorted_idx[:top_k]
    
    # Get the true labels for the top K transactions
    top_k_true = y_true[top_k_idx]
    
    # Get the unique fraudulent card IDs in the top K transactions
    fraudulent_cards_in_top_k = np.unique(top_k_true[top_k_true == 1])
    
    # Get the total unique fraudulent card IDs in the dataset
    total_fraudulent_cards = np.unique(y_true[y_true == 1])
    
    # Calculate Card Precision@K
    card_precision = len(fraudulent_cards_in_top_k) / len(total_fraudulent_cards)
    
    return card_precision


# ### Performance assessment
# The following function computes and plots the AUC ROC, Average Precision, and Card Precision Top K metrics.

def performance_assessment(predictions_df, output_feature='TX_FRAUD', top_k_list=[100]):
    
    # Compute AUC ROC
    auc_roc = roc_auc_score(predictions_df[output_feature], predictions_df.predictions)
    
    # Compute Average Precision
    average_precision = average_precision_score(predictions_df[output_feature], predictions_df.predictions)
    
    # Compute Card Precision Top K
    card_precision_top_k_results = {} 
    for top_k in top_k_list:
        card_precision_top_k_results[top_k] = card_precision_top_k(predictions_df[output_feature], predictions_df.predictions, top_k)
        
    # Create a DataFrame to display the results
    results_df = pd.DataFrame({'AUC ROC': [auc_roc], 
                               'Average precision': [average_precision]})
    
    for top_k, result in card_precision_top_k_results.items():
        results_df[f'Card Precision@{top_k}'] = [result]
        
    print(results_df.round(3).to_markdown(index=False))
    
    # Plot Precision-Recall curve
    plot_pr_curve(predictions_df[output_feature], predictions_df.predictions)


# ## Data preparation functions

# ### Read data from files
# The function below reads the simulated data, in order to create a unique transactions DataFrame.

def read_from_files(DIR_INPUT,
                    start_date,
                    end_date):

    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if f.endswith('.pkl')]
    
    df=pd.DataFrame()
    for file in tqdm(files):
        
        file_date = datetime.datetime.strptime(file.split('/')[-1].split('.')[0], '%Y-%m-%d')
        if (file_date >= start_date) and (file_date <= end_date):
            
            df = pd.concat([df, pd.read_pickle(file)], ignore_index=True)
            
    df=df.sort_values('TX_DATETIME').reset_index(drop=True)
    
    # Using float32 for TX_AMOUNT and others for faster training
    df['TX_AMOUNT']=df.TX_AMOUNT.astype(np.float32)
    df['TX_TIME_SECONDS']=df.TX_TIME_SECONDS.astype(np.int32)
    df['TX_TIME_DAYS']=df.TX_TIME_DAYS.astype(np.int32)
    df['TX_FRAUD']=df.TX_FRAUD.astype(np.int32)
    df['TX_FRAUD_SCENARIO']=df.TX_FRAUD_SCENARIO.astype(np.int32)
    df['TX_DURING_WEEKEND']=df.TX_DURING_WEEKEND.astype(np.int32)
    df['TX_DURING_NIGHT']=df.TX_DURING_NIGHT.astype(np.int32)
    
    return df


# ### Get training and test sets
# The following function creates training and test sets, by splitting the transactions data according to their dates.

def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train, delta_delay, delta_test):
    
    # Get the training set data
    
    start_date_test=start_date_training+datetime.timedelta(days=delta_train)
    
    end_date_training=start_date_training+datetime.timedelta(days=delta_train-1)
    start_date_prediction=start_date_test+datetime.timedelta(days=delta_delay)
    end_date_prediction=start_date_prediction+datetime.timedelta(days=delta_test-1)

    print("Training period: from "+str(start_date_training)+" to "+str(end_date_training))
    print("Prediction period: from "+str(start_date_prediction)+" to "+str(end_date_prediction))
    
    
    # Training set: get all transactions in the training set (day by day data)
    train_df = transactions_df.loc[(transactions_df.TX_DATETIME>=start_date_training) & 
                                            (transactions_df.TX_DATETIME<start_date_test)]
    
    # Test set: get all transactions in the prediction set (day by day data)
    test_df = transactions_df.loc[(transactions_df.TX_DATETIME>=start_date_prediction) & 
                                            (transactions_df.TX_DATETIME<=end_date_prediction)]
    
    return (train_df, test_df)


# ### Scale data
# The following function scales the input features using a StandardScaler.

def scaleData(train_df, test_df, input_features):
    
    scaler = StandardScaler()
    scaler.fit(train_df[input_features])
    
    train_df[input_features] = scaler.transform(train_df[input_features])
    test_df[input_features] = scaler.transform(test_df[input_features])
    
    return (train_df, test_df)


# ## Pytorch helper functions

# ### Seed everything
# The following function sets the random seeds for reproducibility.

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ### Dataset and Dataloader
# The functions below define a Pytorch Dataset, and generate train and validation Dataloaders.

from torch.utils.data import Dataset, DataLoader # Import Dataset and DataLoader here

class FraudDataset(Dataset):

    def __init__(self, x, y=None):
        'Initialization'
        self.x = x
        self.y = y

    def __len__(self):
        'Returns the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample index
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]


def prepare_generators(training_set, valid_set, batch_size=64):

    loader_params = {'batch_size': batch_size,
                     'shuffle': True,
                     'num_workers': 0}

    training_generator = DataLoader(training_set, **loader_params)

    loader_params = {'batch_size': batch_size,
                     'shuffle': False,
                     'num_workers': 0}

    valid_generator = DataLoader(valid_set, **loader_params)
    
    return training_generator, valid_generator


# ### Training Loop
# The `training_loop` function trains a Pytorch model.

def training_loop(model, 
                  training_generator, 
                  valid_generator, 
                  optimizer, 
                  criterion, 
                  max_epochs=100, 
                  verbose=False):

    
    training_start_time = time.time()
    
    train_losses = []
    valid_losses = []
    
    min_valid_loss = np.inf # Changed np.Inf to np.inf
    
    # Patient counter is used to implement early stopping
    patient_counter = 0

    for epoch in range(max_epochs):
        
        start_time = time.time()
        
        # Train
        model.train()
        _train_losses = []
        for x_batch, y_batch in training_generator:
            
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(x_batch)
            
            # Compute Loss
            loss = criterion(y_pred.squeeze(), y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            _train_losses.append(loss.item())
        
        train_losses.append(np.mean(_train_losses))

        # Evaluate
        model.eval()
        _valid_losses = []
        with torch.no_grad():
            for x_batch, y_batch in valid_generator:

                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                # Forward pass
                y_pred = model(x_batch)
                
                # Compute Loss
                loss = criterion(y_pred.squeeze(), y_batch)

                _valid_losses.append(loss.item())
        
        valid_losses.append(np.mean(_valid_losses))
        
        if verbose:
            print(f"Epoch {epoch}: train loss: {train_losses[-1]}\nvalid loss: {valid_losses[-1]}")
        
        # Early stopping
        if valid_losses[-1] < min_valid_loss:
            
            min_valid_loss = valid_losses[-1]
            patient_counter = 0
            
            if verbose:
                print(f"New best score: {min_valid_loss}")
            
        else:
            patient_counter +=1
            if verbose:
                print(f"{patient_counter}  iterations since best score.")
                
            if patient_counter >=5:
                if verbose:
                    print("Early stopping")
                break
                
    training_execution_time = time.time()-training_start_time
    
    return model, training_execution_time, train_losses, valid_losses


# ### Per sample MSE (for autoencoders)
# The `per_sample_mse` function computes the reconstruction error for each transaction.

class FraudDatasetUnsupervised(torch.utils.data.Dataset): # Define this class here for self-containment

    def __init__(self, x,output=True):
        'Initialization'
        self.x = x
        self.output = output

    def __len__(self):
        'Returns the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample index
        item = self.x[index].to(DEVICE)
        if self.output:
            return item, item
        else:
            return item

def per_sample_mse(model,generator):
    model.eval()
    criterion = torch.nn.MSELoss(reduction="none")
    batch_losses = []

    for x_batch, y_batch in generator:
        x_batch = x_batch.to(DEVICE) # Ensure x_batch is on the correct device
        # Forward pass
        y_pred = model(x_batch)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_batch)
        loss_app = list(torch.mean(loss,axis=1).detach().cpu().numpy())
        batch_losses.extend(loss_app)

    return batch_losses
