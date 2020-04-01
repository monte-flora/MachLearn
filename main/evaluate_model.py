import sys
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/machine_learning/build_model') 
from MachLearn import MachLearn

""" usage: stdbuf -oL python evaluate_model.py 2 > & log_evaluate & """
model_names = ['RandomForest', 'XGBoost',  'LogisticRegression']
fcst_time_idx_set = ['first_hour', 'second_hour']
target_vars = [
                'matched_to_tornado_0km',
                'matched_to_severe_wind_0km',
                'matched_to_severe_hail_0km']

fname_params = { }
for model_name in model_names:
    print(f'Current model: {model_name}')
    fname_params['model_name'] = model_name
    for target_var in target_vars:
        print(f'Current target variable: {target_var}')
        fname_params['target_var'] = target_var
        for fcst_time_idx in fcst_time_idx_set:
            fname_params['fcst_time_idx'] = fcst_time_idx
            model = MachLearn(fname_params=fname_params, load_model=True, drop_correlated_features=True)
            model.evaluateCV()



