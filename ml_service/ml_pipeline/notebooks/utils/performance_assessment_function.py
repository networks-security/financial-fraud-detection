from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.append('./utils')  # make sure Python knows where to look

import import_shared_functions as utils_import
import ml_training_functions as utils_training


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

# this assessment function is invoke on the test_set_df (target feature: 'TX_FRAUD') after append all the predicted output from ML (target feature: 'predictions')
def performance_assessment(predictions_df, output_feature='TX_FRAUD', 
                           prediction_feature='predictions', top_k_list=[100],
                           rounded=True):
    
    AUC_ROC = metrics.roc_auc_score(predictions_df[output_feature], predictions_df[prediction_feature])
    AP = metrics.average_precision_score(predictions_df[output_feature], predictions_df[prediction_feature])
    
    performances = pd.DataFrame([[AUC_ROC, AP]], 
                           columns=['AUC ROC','Average precision'])
    
    for top_k in top_k_list:
    
        _, _, mean_card_precision_top_k = card_precision_top_k(predictions_df, top_k)
        performances['Card Precision@'+str(top_k)]=mean_card_precision_top_k
        
    if rounded:
        performances = performances.round(3)
    
    return performances

def performance_assessment_model_collection(fitted_models_and_predictions_dictionary, 
                                            transactions_df, 
                                            type_set='test',
                                            top_k_list=[100]):

    performances=pd.DataFrame() 
    
    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
    
        predictions_df=transactions_df
            
        predictions_df['predictions']=model_and_predictions['predictions_'+type_set]
        
        performances_model=performance_assessment(predictions_df, output_feature='TX_FRAUD', 
                                                   prediction_feature='predictions', top_k_list=top_k_list)
        performances_model.index=[classifier_name]
        
        performances=performances.append(performances_model)
        
    return performances


def execution_times_model_collection(fitted_models_and_predictions_dictionary):

    execution_times=pd.DataFrame() 
    
    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
    
        execution_times_model=pd.DataFrame() 
        execution_times_model['Training execution time']=[model_and_predictions['training_execution_time']]
        execution_times_model['Prediction execution time']=[model_and_predictions['prediction_execution_time']]
        execution_times_model.index=[classifier_name]
        
        execution_times=execution_times.append(execution_times_model)
        
    return execution_times

def get_summary_performances(performances_df, parameter_column_name="Parameters summary"):
    
    performances_results = utils_import.get_summary_performances(performances_df, parameter_column_name)

    metrics = ['AUC ROC','Average precision','Card Precision@100']
    metrics_test = ['AUC ROC Test','Average precision Test','Card Precision@100 Test']

    # Add hyperparameter values from 'Parameters' column for each best result
    best_hyperparam_values = []
    for metric in metrics:
        index_best_validation_performance = performances_df.index[np.argmax(performances_df[metric+' Validation'].values)]
        best_hyperparam_values.append(performances_df['Parameters'].iloc[index_best_validation_performance])
    
    performances_results.loc["Best hyperparameters"]=best_hyperparam_values
    
    optimal_hyperparam_values = []
    for metric in metrics_test:
        index_optimal_test_performance = performances_df.index[np.argmax(performances_df[metric].values)]
        optimal_hyperparam_values.append(performances_df['Parameters'].iloc[index_optimal_test_performance])
    
    performances_results.loc["Optimal hyperparameters"]=optimal_hyperparam_values
    
    return performances_results


def rank_models(model_performances_list, metrics=None, weights=None):

    if isinstance(metrics, str):
        metrics = [metrics]
    
    # Validate weights sum to 1.0
    total_weight = sum(weights.get(m, 0) for m in metrics)
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"Weights must sum to 1.0, but got {total_weight}")
    
    # Extract validation metrics from each model's performance summary
    ranking_data = []
    
    for model_name, perf_df in model_performances_list:
        model_score = {}
        model_score['Model'] = model_name
        
        # Extract validation performance metrics from the summary
        validation_row = perf_df.loc['Validation performance']
        
        for metric in metrics:
            if metric in validation_row.index:
                # Parse metric value (format: "0.87+/-0.01")
                metric_value = validation_row[metric]
                if isinstance(metric_value, str):
                    metric_value = float(metric_value.split('+/-')[0])
                model_score[metric] = metric_value
            else:
                raise ValueError(f"Metric '{metric}' not found in performance summary")
        
        # Calculate weighted score if multiple metrics
        if len(metrics) > 1:
            weighted_score = sum(
                model_score[metric] * weights.get(metric, 0) 
                for metric in metrics
            )
            model_score['Weighted Score'] = weighted_score
        else:
            model_score['Score'] = model_score[metrics[0]]
        
        ranking_data.append(model_score)
    
    # Create ranking dataframe
    ranking_df = pd.DataFrame(ranking_data)
    
    # Sort by score (descending - higher is better)
    score_column = 'Weighted Score' if len(metrics) > 1 else 'Score'
    ranking_df = ranking_df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
    
    # # Add rank column
    # ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
    
    return ranking_df


# for this function, the transactions_df_scorer must contain CUSTOMER_ID and TX_TIME_DAYS for the full df; X_index are the indices for the current test fold.
def card_precision_top_k_wrapper(probs, X_index, transactions_df_scorer, k=100):
    preds_df = transactions_df_scorer.loc[X_index].copy()
    preds_df['predictions'] = probs
    nb, per_day_list, mean_cp = utils_training.card_precision_top_k(preds_df, k)
    return mean_cp

def get_performance_metrics(df, y, probs, transactions_df_scorer):
    auc = roc_auc_score(y, probs)
    ap  = average_precision_score(y, probs)
    cp = None
    if transactions_df_scorer is not None:
        cp = card_precision_top_k_wrapper(probs, df.index, transactions_df_scorer, k=100)
        
    return auc, ap, cp