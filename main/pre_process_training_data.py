import sys
sys.path.append('/home/monte.flora/machine_learning/build_model') 
from PreProcess import PreProcess 
import pickle

""" usage: stdbuf -oL python pre_process_training_data.py  2 > & log_preprocess & """
model = PreProcess( )

TRAINING = 'training'
VALIDATION = 'validation'
TESTING = 'testing'
target_vars = [  
                'matched_to_tornado_0km',
                'matched_to_severe_wind_0km',
                'matched_to_severe_hail_0km']

for target_var in target_vars:
    for fcst_time_idx in [ 'first_hour', 'second_hour' ]:
            print( '-----------------------------------------')
            print(f' Target Var: {target_var} ')
            print(f' FCST_TIME_IDX: {fcst_time_idx} ' )
            print( '-----------------------------------------')
            fname_params = {
                        'target_var': target_var,
                        'fcst_time_idx': fcst_time_idx,
                            }
            model.preprocess( 
                              fname_params=fname_params, 
                              modes = [TRAINING, VALIDATION, TESTING],
                              load_filename_template = '{}_f:{}_t:{}_raw_probability_objects.pkl' 
                              )

    
        
        

