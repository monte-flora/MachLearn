import sys
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/machine_learning/build_model') 
from MachLearn import MachLearn
from feature_names import _feature_names_for_traditional_ml
import feature_names as fn
import numpy as np 
import pandas as pd
import pickle
from os.path import join

""" usage: stdbuf -oL python valid_model.py 2 > & log_valid & """

target_vars = [
                'matched_to_tornado_0km',
                'matched_to_severe_wind_0km',
                'matched_to_severe_hail_0km']
fname_params = { }

def return_params( model_name ):
    if model_name == 'RandomForest':
             #RandomForest Grid
            param_grid = { 'bootstrap': [True],
               'max_depth': [10, 20, 40, None],
               'n_estimators': [100, 300, 500, 750],
               'max_features': ['log2'],
               'min_samples_leaf': [5, 15, 25, 50],
               'criterion' : ['entropy']
               }    
    if model_name == "XGBoost":
            param_grid = {
                   'nrounds': [100, 200, 500, 750],
                   'max_depth': [6,8,10,15],
                   'eta': [0.001, 0.01, 0.1],
                   'objective': ['binary:logistic'],
                   'eval_metric': ['aucpr'],
                   'colsample_bytree': [0.8, 1.0],
                   'subsample': [0.8, 1.0],
                   'random_state':[0],
                   'n_jobs': [40]
                } 

    if model_name == "LogisticRegression":
        param_grid = {
                'l1_ratio': [0.01, 0.5, 1.0],
                'C': [0.01, 0.1, 0.5, 1.0, 2.0]
                } 

    return param_grid

path = '/home/monte.flora/machine_learning/main/model_parameters/'

for model_name in ['RandomForest', 'XGBoost', 'LogisticRegression']:
    print(f'Current model: {model_name}')
    fname_params['model_name'] = model_name
    param_grid = return_params(model_name) 
    for target_var in target_vars:    
        print(f'Current target variable: {target_var}')
        fname_params['target_var'] = target_var
        fname_params['fcst_time_idx'] = 'first_hour'
        model = MachLearn(fname_params=fname_params, drop_correlated_features=True) 
        best_params, avg_scores = model.hyperparameter_search(param_grid=param_grid)
        print('Best Parameters for {}: {}'.format(fname_params['model_name'], best_params))
        print('AUPRC across the validation folds: {}'.format(avg_scores))
   
        save_name = join(path, f'best_{fname_params["model_name"]}_{fname_params["target_var"]}_params.pkl')
        with open( save_name, 'wb') as fp:
            pickle.dump(best_params, fp)
    
