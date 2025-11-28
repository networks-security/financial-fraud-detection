import time
import datetime
import pandas as pd
import numpy as np
import sklearn
import imblearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import os
import math
import json
import random

import sys
sys.path.append('./utils')  # make sure Python knows where to look

import transform_data_functions as utils_transform_data
import performance_assessment_function as utils_assessment

# NumPy compatibility patch for deprecated np.int (removed in NumPy 1.20+)
# This fixes issues with older libraries like mlens that use deprecated aliases
if not hasattr(np, 'int'):
    np.int = np.int64
    np.float = np.float64
    np.complex = np.complex128
    np.object = np.object_
    np.str = np.str_
    np.long = np.int64
    np.unicode = np.str_


def scaleData(train,test,features):
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train[features])
    train[features]=scaler.transform(train[features])
    test[features]=scaler.transform(test[features])
    
    return (train,test)

def fit_model_and_get_predictions(classifier, train_df, test_df, 
                                  input_features, output_feature="TX_FRAUD",scale=True):

    # scales/standardized dataset base on input features
    if scale:
        (train_df, test_df)=scaleData(train_df,test_df,input_features)
    
    # We first train the classifier using the `fit` method, and pass as arguments the input and output features
    start_time=time.time()
    classifier.fit(train_df[input_features], train_df[output_feature])
    training_execution_time=time.time()-start_time

    # We then get the predictions on the training and test data using the `predict_proba` method
    # The predictions are returned as a numpy array, that provides the probability of fraud for each transaction 
    start_time=time.time()
    predictions_test=classifier.predict_proba(test_df[input_features])[:,1]
    prediction_execution_time=time.time()-start_time
    
    predictions_train=classifier.predict_proba(train_df[input_features])[:,1]

    # The result is returned as a dictionary containing the fitted models, 
    # and the predictions on the training and test sets
    model_and_predictions_dictionary = {'classifier': classifier,
                                        'predictions_train': predictions_train,
                                        'predictions_test': predictions_test,
                                        'training_execution_time': training_execution_time,
                                        'prediction_execution_time': prediction_execution_time
                                       }
    
    return model_and_predictions_dictionary

# ----------------------------- Implementation on CV and Hyperparam Tuning ----------------------------- #

def prequentialSplit(transactions_df,
                    start_date_training, 
                    n_folds=4, 
                    delta_train=7,
                    delta_delay=7,
                    delta_assessment=7):
    
    prequential_split_indices=[]
        
    # For each fold
    for fold in range(n_folds):
        
        # Shift back start date for training by the fold index times the assessment period (delta_assessment)
        # (See Fig. 5)
        start_date_training_fold = start_date_training-datetime.timedelta(days=fold*delta_assessment)
        
        # Get the training and test (assessment) sets
        (train_df, test_df)=utils_transform_data.get_train_test_set(transactions_df,
                                               start_date_training=start_date_training_fold,
                                               delta_train=delta_train,delta_delay=delta_delay,delta_test=delta_assessment)
    
        # Get the indices from the two sets, and add them to the list of prequential splits
        indices_train=list(train_df.index)
        indices_test=list(test_df.index)
        
        prequential_split_indices.append((indices_train,indices_test))
    
    return prequential_split_indices


def card_precision_top_k_day(df_day,top_k):
    
    # This takes the max of the predictions AND the max of label TX_FRAUD for each CUSTOMER_ID, 
    # and sorts by decreasing order of fraudulent prediction
    df_day = df_day.groupby('CUSTOMER_ID').max().sort_values(by="predictions", ascending=False).reset_index(drop=False)
            
    # Get the top k most suspicious cards
    df_day_top_k=df_day.head(top_k)
    list_detected_compromised_cards=list(df_day_top_k[df_day_top_k.TX_FRAUD==1].CUSTOMER_ID)
    
    # Compute precision top k
    card_precision_top_k = len(list_detected_compromised_cards) / top_k
    
    return list_detected_compromised_cards, card_precision_top_k


def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards=True):

    # Sort days by increasing order
    list_days=list(predictions_df['TX_TIME_DAYS'].unique())
    list_days.sort()
    
    # At first, the list of detected compromised cards is empty
    list_detected_compromised_cards = []
    
    card_precision_top_k_per_day_list = []
    nb_compromised_cards_per_day = []
    
    # For each day, compute precision top k
    for day in list_days:
        
        df_day = predictions_df[predictions_df['TX_TIME_DAYS']==day]
        df_day = df_day[['predictions', 'CUSTOMER_ID', 'TX_FRAUD']]
        
        # Let us remove detected compromised cards from the set of daily transactions
        df_day = df_day[df_day.CUSTOMER_ID.isin(list_detected_compromised_cards)==False]
        
        nb_compromised_cards_per_day.append(len(df_day[df_day.TX_FRAUD==1].CUSTOMER_ID.unique()))
        
        detected_compromised_cards, card_precision_top_k = card_precision_top_k_day(df_day,top_k)
        
        card_precision_top_k_per_day_list.append(card_precision_top_k)
        
        # Let us update the list of detected compromised cards
        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_compromised_cards)
        
    # Compute the mean
    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()
    
    # Returns precision top k per day as a list, and resulting mean
    return nb_compromised_cards_per_day,card_precision_top_k_per_day_list,mean_card_precision_top_k


def card_precision_top_k_custom(y_true, y_pred, top_k, transactions_df):
    
    # Let us create a predictions_df DataFrame, that contains all transactions matching the indices of the current fold
    # (indices of the y_true vector)
    predictions_df=transactions_df.iloc[y_true.index.values].copy()
    predictions_df['predictions']=y_pred
    
    # Compute the CP@k using the function implemented in Chapter 4, Section 4.2
    nb_compromised_cards_per_day,card_precision_top_k_per_day_list,mean_card_precision_top_k=card_precision_top_k(predictions_df, top_k)
    
    # Return the mean_card_precision_top_k
    return mean_card_precision_top_k


## ----------------------------- Implementation on Different CV search technique for hyperparam tuning ----------------------------- #

def prequential_parameters_search(transactions_df, 
                            classifier, 
                            input_features, output_feature, 
                            parameters, scoring, cv,
                            start_date_training, 
                            n_folds=4,
                            expe_type='Test',
                            delta_train=7, 
                            delta_delay=7, 
                            delta_assessment=7,
                            performance_metrics_list_grid=['roc_auc'],
                            performance_metrics_list=['AUC ROC'],
                            type_search='grid',
                            n_iter=10,
                            random_state=0,
                            n_jobs=-1):
    
    scaler = [('scaler', sklearn.preprocessing.StandardScaler())]
    clf = [('clf', classifier)]
    estimators = scaler + clf
    pipe = sklearn.pipeline.Pipeline(estimators)
    
    # prequential_split_indices=prequentialSplit(transactions_df,
    #                                            start_date_training=start_date_training, 
    #                                            n_folds=n_folds, 
    #                                            delta_train=delta_train, 
    #                                            delta_delay=delta_delay, 
    #                                            delta_assessment=delta_assessment)
    
    parameters_search = None
    
    if type_search=="grid":
        
        parameters_search = GridSearchCV(pipe, parameters, scoring=scoring, cv=cv, 
                                         refit=False, n_jobs=n_jobs, error_score='raise')
    
    if type_search=="random":
        
        parameters_search = RandomizedSearchCV(pipe, parameters, scoring=scoring, cv=cv, 
                                     refit=False, n_jobs=n_jobs,n_iter=n_iter,random_state=random_state, error_score='raise')

    
    X=transactions_df[input_features]
    y=transactions_df[output_feature]

    parameters_search.fit(X, y)
    # best_params = parameters_search.best_params_
    
    performances_df=pd.DataFrame()
    
    for i in range(len(performance_metrics_list_grid)):
        performances_df[performance_metrics_list[i]+' '+expe_type]=parameters_search.cv_results_['mean_test_'+performance_metrics_list_grid[i]]
        performances_df[performance_metrics_list[i]+' '+expe_type+' Std']=parameters_search.cv_results_['std_test_'+performance_metrics_list_grid[i]]

    performances_df['Parameters']=parameters_search.cv_results_['params']
    performances_df['Execution time']=parameters_search.cv_results_['mean_fit_time']
    
    return performances_df

def prequential_parameters_search_with_sample(transactions_df, 
                            classifier, sampler_list,
                            input_features, output_feature, 
                            parameters, scoring, cv,
                            start_date_training, 
                            n_folds=4,
                            expe_type='Test',
                            delta_train=7, 
                            delta_delay=7, 
                            delta_assessment=7,
                            performance_metrics_list_grid=['roc_auc'],
                            performance_metrics_list=['AUC ROC'],
                            type_search='grid',
                            n_iter=10,
                            random_state=0,
                            n_jobs=-1):
    
    
    # prequential_split_indices=prequentialSplit(transactions_df,
    #                                            start_date_training=start_date_training, 
    #                                            n_folds=n_folds, 
    #                                            delta_train=delta_train, 
    #                                            delta_delay=delta_delay, 
    #                                            delta_assessment=delta_assessment)
    
    scaler = [('scaler', sklearn.preprocessing.StandardScaler())]
    clf = [('clf', classifier)]
    estimators = sampler_list + scaler + clf
    pipe = imblearn.pipeline.Pipeline(estimators)

    parameters_search = None
    
    if type_search=="grid":
        parameters_search = GridSearchCV(pipe, parameters, scoring=scoring, cv=cv, 
                                         refit=False, n_jobs=n_jobs)
    
    if type_search=="random":
        parameters_search = RandomizedSearchCV(pipe, parameters, scoring=scoring, cv=cv, 
                                     refit=False, n_jobs=n_jobs,n_iter=n_iter,random_state=random_state)

    
    X=transactions_df[input_features]
    y=transactions_df[output_feature]

    parameters_search.fit(X, y)
    # best_params = parameters_search.best_params_
    
    performances_df=pd.DataFrame()
    
    for i in range(len(performance_metrics_list_grid)):
        performances_df[performance_metrics_list[i]+' '+expe_type]=parameters_search.cv_results_['mean_test_'+performance_metrics_list_grid[i]]
        performances_df[performance_metrics_list[i]+' '+expe_type+' Std']=parameters_search.cv_results_['std_test_'+performance_metrics_list_grid[i]]

    performances_df['Parameters']=parameters_search.cv_results_['params']
    performances_df['Execution time']=parameters_search.cv_results_['mean_fit_time']
    
    return performances_df

def model_selection_wrapper(transactions_df, 
                            classifier, 
                            input_features, output_feature,
                            parameters, 
                            scoring, cv,
                            start_date_training_for_valid,
                            start_date_training_for_test,
                            n_folds=4,
                            delta_train=7, 
                            delta_delay=7, 
                            delta_assessment=7,
                            performance_metrics_list_grid=['roc_auc'],
                            performance_metrics_list=['AUC ROC'],
                            type_search='grid',
                            n_iter=10,
                            random_state=0,
                            n_jobs=-1):

    # Get performances on the validation set using prequential validation
    performances_df_validation=prequential_parameters_search(
                            transactions_df, classifier, 
                            input_features, output_feature,
                            parameters, scoring, cv,
                            start_date_training=start_date_training_for_valid,
                            n_folds=n_folds,
                            expe_type='Validation',
                            delta_train=delta_train, 
                            delta_delay=delta_delay, 
                            delta_assessment=delta_assessment,
                            performance_metrics_list_grid=performance_metrics_list_grid,
                            performance_metrics_list=performance_metrics_list,
                            type_search=type_search,
                            n_iter=n_iter,
                            random_state=random_state,
                            n_jobs=n_jobs)
    
    # Get performances on the test set using prequential validation
    performances_df_test=prequential_parameters_search(
                            transactions_df, classifier, 
                            input_features, output_feature,
                            parameters, scoring, cv, 
                            start_date_training=start_date_training_for_test,
                            n_folds=n_folds,
                            expe_type='Test',
                            delta_train=delta_train, 
                            delta_delay=delta_delay, 
                            delta_assessment=delta_assessment,
                            performance_metrics_list_grid=performance_metrics_list_grid,
                            performance_metrics_list=performance_metrics_list,
                            type_search=type_search,
                            n_iter=n_iter,
                            random_state=random_state,
                            n_jobs=n_jobs)
    
    # Bind the two resulting DataFrames
    performances_df_validation.drop(columns=['Parameters','Execution time'], inplace=True)
    performances_df=pd.concat([performances_df_test,performances_df_validation],axis=1)

    # And return as a single DataFrame
    return performances_df

def model_selection_wrapper_with_sample(transactions_df, 
                            classifier, sampler_list,
                            input_features, output_feature,
                            parameters, 
                            scoring, cv, 
                            start_date_training_for_valid,
                            start_date_training_for_test,
                            n_folds=4,
                            delta_train=7, 
                            delta_delay=7, 
                            delta_assessment=7,
                            performance_metrics_list_grid=['roc_auc'],
                            performance_metrics_list=['AUC ROC'],
                            type_search='grid',
                            n_iter=10,
                            random_state=0,
                            n_jobs=-1):

    # Get performances on the validation set using prequential validation
    performances_df_validation=prequential_parameters_search_with_sample(
                            transactions_df, 
                            classifier, sampler_list,
                            input_features, output_feature,
                            parameters, scoring, cv, 
                            start_date_training=start_date_training_for_valid,
                            n_folds=n_folds,
                            expe_type='Validation',
                            delta_train=delta_train, 
                            delta_delay=delta_delay, 
                            delta_assessment=delta_assessment,
                            performance_metrics_list_grid=performance_metrics_list_grid,
                            performance_metrics_list=performance_metrics_list,
                            type_search=type_search,
                            n_iter=n_iter,
                            random_state=random_state,
                            n_jobs=n_jobs)
    
    # Get performances on the test set using prequential validation
    performances_df_test=prequential_parameters_search_with_sample(
                            transactions_df, 
                            classifier, sampler_list,
                            input_features, output_feature,
                            parameters, scoring, cv,
                            start_date_training=start_date_training_for_test,
                            n_folds=n_folds,
                            expe_type='Test',
                            delta_train=delta_train, 
                            delta_delay=delta_delay, 
                            delta_assessment=delta_assessment,
                            performance_metrics_list_grid=performance_metrics_list_grid,
                            performance_metrics_list=performance_metrics_list,
                            type_search=type_search,
                            n_iter=n_iter,
                            random_state=random_state,
                            n_jobs=n_jobs)
    
    # Bind the two resulting DataFrames
    performances_df_validation.drop(columns=['Parameters','Execution time'], inplace=True)
    performances_df=pd.concat([performances_df_test,performances_df_validation],axis=1)

    # And return as a single DataFrame
    return performances_df

def get_predict_proba(model, X_scaled):
    if hasattr(model, "predict_proba"): # some models might not implement predict_proba
        return model.predict_proba(X_scaled)[:, 1]
    elif hasattr(model, "decision_function"):  # if model implement decision_function -> convert to probabilities via logistic/sigmoid
        dec = model.decision_function(X_scaled)
        return 1.0 / (1.0 + np.exp(-dec))  # sigmoid to convert to probability-like
    else: # last resort: use predict (not recommended)
        return model.predict(X_scaled).astype(float)

def get_super_learner_prediction(transformed_transaction_df, load_model):
    base_models = load_model['base_models']
    meta_model = load_model['meta_model']
    scaler = load_model['scaler']
    input_features = load_model['input_features']
    
    X = transformed_transaction_df[input_features].values
    X_scaled = scaler.transform(X)

    # Add a dummy 16th feature to match model training if input_features only has 15 (but base models were trained with 16 features)
    first_model = next(iter(base_models.values()))
    if hasattr(first_model, 'n_features_in_'):
        expected_features = first_model.n_features_in_
        if X_scaled.shape[1] < expected_features:
            # Add dummy columns filled with 0
            dummy_cols = np.zeros((X_scaled.shape[0], expected_features - X_scaled.shape[1]))
            X_scaled = np.hstack([X_scaled, dummy_cols])

    # Base model probabilities
    probs_list = []
    for _, model in base_models.items():
        prob = get_predict_proba(model, X_scaled)
        probs_list.append(prob.reshape(-1, 1))
    meta_X = np.hstack(probs_list)

    # Final prediction probability
    return get_predict_proba(meta_model, meta_X)