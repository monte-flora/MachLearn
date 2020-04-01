import sys
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/machine_learning/build_model') 
from MachLearn import MachLearn
from feature_names import _feature_names_for_traditional_ml
import feature_names as fn

""" usage: stdbuf -oL python predict.py 2 > & log_predict & """
debug = True 
model = MachLearn( )
fname_params = {
           'model_name': 'RandomForest',
           'target_var': 'matched_to_tornado_0km',
           'resampling_method': 'random',
           'fcst_time_idx': 'first_hour',
           'time_idx': 6
           }

removed_variables = [
                    'matched_to_severe_wx_warn_polys_15km', 'matched_to_severe_wx_warn_polys_30km',
                    'matched_to_tornado_warn_ploys_15km', 'matched_to_tornado_warn_ploys_30km']

dates = config.ml_dates
times = config.verification_times
indexs = range(0, 24+1)

pbar = tqdm( total =len(list(to_iterator(dates,times,indexs))))
run_parallel(
            func = worker,
            nprocs_to_use = 0.3,
            iterator = to_iterator(dates,times,indexs)
            )

model.predict_probaCV(
                      fname_params=fname_params, 
                      variables_to_remove = removed_variables,
                      debug=debug
                     )



