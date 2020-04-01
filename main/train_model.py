import sys

sys.path.append("/home/monte.flora/wofs/util")
sys.path.append("/home/monte.flora/machine_learning/build_model")
from MachLearn import MachLearn
from feature_names import _feature_names_for_traditional_ml
import feature_names as fn
import numpy as np
import pandas as pd
import pickle
from os.path import join

""" usage: stdbuf -oL python train_model.py 2 > & log_train & """
fold_to_load = None

"""
model_set = ['RandomForest']
fcst_time_idx_set = ['first_hour']
target_vars = ['matched_to_tornado_0km']
correlated_feature_set = [True, False]
"""

model_set = ['RandomForest', 'XGBoost', "LogisticRegression"]
fcst_time_idx_set = ["first_hour", "second_hour"]
target_vars = [
    "matched_to_tornado_0km",
    "matched_to_severe_wind_0km",
    "matched_to_severe_hail_0km",
]
correlated_feature_set = [False]

def return_params(fname_params):
    path = "/home/monte.flora/machine_learning/main/model_parameters/"
    save_name = join(
        path,
        f'best_{fname_params["model_name"]}_{fname_params["target_var"]}_params.pkl',
    )
    with open(save_name, "rb") as pkl_file:
        params = pickle.load(pkl_file)
    return params

fname_params = {}
for drop_correlated_features in correlated_feature_set:
    for fcst_time_idx in fcst_time_idx_set:
        fname_params["fcst_time_idx"] = fcst_time_idx
        for model_name in model_set:
            fname_params["model_name"] = model_name
            for target_var in target_vars:
                fname_params["target_var"] = target_var
                params = return_params(fname_params)
                print("----------------------------------------------------------------------")
                print(f"Current model: {model_name} and parameters: {params}")  
                print(f"Current target variable: {target_var}")
                print("----------------------------------------------------------------------")
                model = MachLearn(
                    fname_params=fname_params,
                    load_model=False,
                    drop_correlated_features=drop_correlated_features,
                    fold_to_load = fold_to_load
                )
                model.fitCV(model_params=params)
