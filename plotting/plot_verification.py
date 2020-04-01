import argparse
import os
# Setting up the parser 
ml_model_names = ['RandomForest', 'XGBoost', 'LogisticRegression']
fcst_time_idx_set = ['first_hour', 'second_hour'] #[0,6,12,18, 24,'first_hour','second_hour']
target_vars = [
                'matched_to_tornado_0km',
                'matched_to_severe_wind_0km',
                'matched_to_severe_hail_0km']

list_of_scripts = ['plot_perform.py',
                   'plot_roc.py', 
                   'plot_reliability.py'
                   ]

correlated_feature_opt = 0

for verify_var in target_vars:
    for fcst_time_idx in fcst_time_idx_set:
        for script in list_of_scripts:
            cmd = f'python {script} -m {ml_model_names} -t {fcst_time_idx} --target {verify_var} -c {correlated_feature_opt} &'
            print( f'Executing {cmd}...' )
            os.system(cmd)

