from os.path import join
import xarray as xr
from wofs.util import config

def load_dataset( model_name, fcst_time_idx, vars_to_load, verify_var, correlated_feature_opt ):    
    '''Loads results of the verification
    '''
    if correlated_feature_opt:
        tag = 'correlated_features_removed'
    else:
        tag = 'all_features'

    fname = join( config.ML_RESULTS_PATH, f'verifyData_{model_name}_target:{verify_var}_fcst_time_idx={fcst_time_idx}_{tag}.nc')
    ds = xr.open_dataset( fname ) 
    return {name: ds[name].values for name in vars_to_load}

def load_different_models( ml_model_names, fcst_time_idx, verify_var, vars_to_load, correlated_feature_opt):
    '''
    Loads model results from multiple ML models at a single lead time
    '''
    results = {name: load_dataset( model_name = name,
                            fcst_time_idx = fcst_time_idx,
                            verify_var = verify_var,
                            vars_to_load=vars_to_load,
                            correlated_feature_opt=correlated_feature_opt)
                            for name in ml_model_names}
    return results

def load_different_times( model_name, fcst_time_idx_set, verify_var, vars_to_load, correlated_feature_opt):
    '''
    Loads model results from a single ML model at different lead times
    '''
    duration = 30 
    results = {f'{model_name}': [ load_dataset( model_name = model_name,
                            fcst_time_idx = t,
                            verify_var = verify_var,
                            vars_to_load=vars_to_load,
                            correlated_feature_opt=correlated_feature_opt)
                            for t in fcst_time_idx_set]}
    return results

def load_different_features( model_name, fcst_time_idx, verify_var, vars_to_load, correlated_feature_opt_set):
    '''
    Loads model results from a single ML model at different lead times
    '''
    corr_dict = {True:  'correlated_features_removed', False:  'all_features'}
    results = {f'{model_name} {corr_dict[correlated_feature_opt]}': load_dataset( model_name = model_name,
                            fcst_time_idx = fcst_time_idx,
                            verify_var = verify_var,
                            vars_to_load=vars_to_load,
                            correlated_feature_opt=correlated_feature_opt)
                            for correlated_feature_opt in correlated_feature_opt_set}
    return results

###########################################################################################

def load( adict ):
    if adict['mode'][0] == 'different_models':
        results = load_different_models( ml_model_names = adict['mode'][1],
                        fcst_time_idx = adict['fcst_time_idx'],
                        verify_var = adict['verify_var'],
                        vars_to_load = adict['vars_to_load'],
                        correlated_feature_opt = adict['correlated_feature_opt']
                        )
    
    elif adict['mode'][0] == 'different_times':
       results = load_different_times(  model_name = adict['model_name'],
                        fcst_time_idx_set = adict['mode'][1],
                        verify_var = adict['verify_var'],
                        vars_to_load = adict['vars_to_load'],
                        correlated_feature_opt = adict['correlated_feature_opt']
                      )
    
    elif adict['mode'][0] == 'different_features':
        results = load_different_features(  model_name = adict['model_name'],
                        fcst_time_idx = adict['fcst_time_idx'],
                        verify_var = adict['verify_var'],
                        vars_to_load = adict['vars_to_load'],
                        correlated_feature_opt_set = adict['mode'][1]
                      ) 
        
    return results

