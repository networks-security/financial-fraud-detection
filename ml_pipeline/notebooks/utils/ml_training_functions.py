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
import performance_assessment_function as utils_ml_assessment


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

# # ---------------------------- Different Classifier and its Param ----------------------------

# import sklearn
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier


# # this handle imbalance dataset with 'class_weight' param
# logreg_clf = sklearn.linear_model.LogisticRegression() # -> based on param
# logreg_params = {'clf__C':[0.1,1,10,100], 
#               'clf__class_weight':['balanced'],
#               'clf__random_state':[0]}
# # this handle imbalance dataset with hybrid resampling technique
# sampler_list = [('sampler1', imblearn.over_sampling.SMOTE()),
#                 ('sampler2', imblearn.under_sampling.RandomUnderSampler())]
# logreg_params_hybridsampling = {'clf__C':[0.1,1,10,100], 'clf__random_state':[0],
#               'sampler1__sampling_strategy':[0.1], 
#               'sampler2__sampling_strategy':[0.1, 0.5, 1], 
#               'sampler1__random_state':[0], 'sampler2__random_state':[0]}
# # this handle imbalance dataset with only under-resampling technique
# sampler_list = [('sampler1', imblearn.over_sampling.SMOTE())]
# logreg_params_undersampling = {'clf__C':[0.1,1,10,100], 'clf__random_state':[0],
#               'sampler1__sampling_strategy':[0.1],
#               'sampler1__random_state':[0]}

# dt_clf = imblearn.ensemble.BalancedBaggingClassifier()
# # Set of parameters for which to assess model performances
# dt_params = {'clf__base_estimator':[sklearn.tree.DecisionTreeClassifier(max_depth=20,random_state=0)], 
#               'clf__n_estimators':[2,3,4,5,6,7,8,9,10,20,50,100],
#               'clf__sampling_strategy':[0.02, 0.05, 0.1, 0.5, 1], 
#               'clf__bootstrap':[True],
#               'clf__sampler':[imblearn.under_sampling.RandomUnderSampler()],
#               'clf__random_state':[0],
#               'clf__n_jobs':[-1]}

# brf_clf = imblearn.ensemble.BalancedRandomForestClassifier()
# brf_params = {'clf__max_depth':[5,10,20,50], 
#               'clf__n_estimators':[25,50,100], 
#               'clf__sampling_strategy':[0.01, 0.05, 0.1, 0.5, 1], 
#               'clf__random_state':[0],
#               'clf__n_jobs':[-1],
#               'clf__random_state':[0], 
#               'clf__n_jobs':[-1]}

# gb_clf = GradientBoostingClassifier()
# # this handle imbalance dataset with hybrid resampling technique
# sampler_list = [('sampler1', imblearn.over_sampling.SMOTE()),
#                 ('sampler2', imblearn.under_sampling.RandomUnderSampler())]
# gb_params_hybridsampling = {'clf__n_estimators': [50, 100, 200],
#             'clf__max_depth': [2, 3, 5],
#             'clf__learning_rate': [0.01, 0.05, 0.1],
#             'clf__n_jobs':[-1], 
#             'clf__random_state':[0],
#             'sampler1__sampling_strategy':[0.1], 
#             'sampler2__sampling_strategy':[0.1, 0.5, 1], 
#             'sampler1__random_state':[0], 'sampler2__random_state':[0]}
# # this handle imbalance dataset with only under-resampling technique
# sampler_list = [('sampler1', imblearn.over_sampling.SMOTE())]
# gb_params_undersampling = {'clf__n_estimators': [50, 100, 200],
#             'clf__max_depth': [2, 3, 5],
#             'clf__learning_rate': [0.01, 0.05, 0.1],
#             'clf__n_jobs':[-1], 
#             'clf__random_state':[0],
#             'sampler1__sampling_strategy':[0.1],
#             'sampler1__random_state':[0]}

# lgbm_clf = lgb.LGBMClassifier()
# lgbm_params = {'clf__max_depth':[3,6,9], 
#               'clf__n_estimators':[25,50,100], 
#               'clf__learning_rate':[0.1,0.3], 
#               'clf__num_leaves': [15, 31, 63, 127],
#               'clf__subsample': [0.7, 0.8, 1.0],
#               'clf__scale_pos_weight':[1,5,10,50,100], # Option 1 to handle imbalance dataset
#               'clf__is_unbalance': [True, False], # Option 2 to handle imbalance dataset
#               'clf__random_state':[0], 
#               'clf__n_jobs':[-1]}

# xgb_clf = xgboost.XGBClassifier()
# xgb_params = {'clf__max_depth':[3,6,9], 
#               'clf__n_estimators':[25,50,100], 
#               'clf__learning_rate':[0.1,0.3], 
#               'clf__scale_pos_weight':[1,5,10,50,100], 
#               'clf__random_state':[0], 
#               'clf__n_jobs':[-1]}

# cat_clf = CatBoostClassifier()
# cat_params = {
#     'iterations': [100, 200, 500],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'depth': [4, 6, 8],
#     'l2_leaf_reg': [1, 3, 5, 7],
#     'bagging_temperature': [0, 1, 5],
#     'class_weights': [
#         None,
#         [1, 10],
#         [1, 20],
#         [1, 50]
#     ]
# }

# start_time=time.time()

# performances_df=model_selection_wrapper(transactions_df, classifier, 
#                                         input_features, output_feature,
#                                         parameters, scoring, 
#                                         start_date_training_for_valid,
#                                         start_date_training_for_test,
#                                         n_folds=n_folds,
#                                         delta_train=delta_train, 
#                                         delta_delay=delta_delay, 
#                                         delta_assessment=delta_assessment,
#                                         performance_metrics_list_grid=performance_metrics_list_grid,
#                                         performance_metrics_list=performance_metrics_list,
#                                         type_search='random',
#                                         n_iter=10,
#                                         random_state=0,
#                                         n_jobs=1)

# execution_time_boosting_random = time.time()-start_time

# parameters_dict=dict(performances_df['Parameters'])
# performances_df['Parameters summary']=[str(parameters_dict[i]['clf__n_estimators'])+
#                                    '/'+ str(parameters_dict[i]['clf__learning_rate'])+
#                                    '/'+ str(parameters_dict[i]['clf__max_depth'])
#                                    for i in range(len(parameters_dict))]

# # Rename to performances_df_xgboost_random for model performance comparison
# performances_df_xgboost_random=performances_df


# # ---------------------------- Apply hyperparam tuning with combined sampling techniques ----------------------------



# # Define sampling strategy
# sampler_list = [('sampler1', imblearn.over_sampling.SMOTE()),
#                 ('sampler2', imblearn.under_sampling.RandomUnderSampler())
#                ]

# # Define classifier
# classifier = sklearn.tree.DecisionTreeClassifier()
# parameters = {'clf__max_depth':[5], 'clf__random_state':[0],
#               'sampler1__sampling_strategy':[0.1], 
#               'sampler2__sampling_strategy':[0.1, 0.5, 1], 
#               'sampler1__random_state':[0], 'sampler2__random_state':[0]}



# start_time = time.time()

# # Fit models and assess performances for all parameters
# performances_df = model_selection_wrapper_with_sampler(transactions_df, classifier, sampler_list, 
#                                                      input_features, output_feature,
#                                                      parameters, scoring, 
#                                                      start_date_training_for_valid,
#                                                      start_date_training_for_test,
#                                                      n_folds=n_folds,
#                                                      delta_train=delta_train, 
#                                                      delta_delay=delta_delay, 
#                                                      delta_assessment=delta_assessment,
#                                                      performance_metrics_list_grid=performance_metrics_list_grid,
#                                                      performance_metrics_list=performance_metrics_list,
#                                                      n_jobs=1)

# execution_time_dt_combined = time.time()-start_time

# # Select parameter of interest (max_depth)
# parameters_dict = dict(performances_df['Parameters'])
# performances_df['Parameters summary']=[parameters_dict[i]['sampler2__sampling_strategy'] for i in range(len(parameters_dict))]

# # Rename to performances_df_combined for model performance comparison at the end of this notebook
# performances_df_combined = performances_df