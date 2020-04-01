import numpy as np
import itertools
import xarray as xr 
from os.path import join, exists
import os 
import sys
from datetime import datetime

#personal modules 
from wofs.data.loadEnsembleData import EnsembleData
from wofs.data.loadEnsembleData import calc_time_max, calc_time_tendency
from wofs.data.loadMRMSData import MRMSData
from wofs.util import config 
from machine_learning.extraction.StormBasedFeatureEngineering import StormBasedFeatureEngineering
from wofs.util.MultiProcessing import multiprocessing_per_date
from wofs.util.feature_names import _feature_names_for_traditional_ml
import wofs.util.feature_names as fn 
from wofs.util.basic_functions import convert_to_seconds, personal_datetime

""" usage: stdbuf -oL python -u ml_feature_extraction.py 2 > & log_extract & """
debug = False

variable_key = 'updraft' 
extraction_method = 'hagelslag' # hagelslag, inflow_sector
extract = StormBasedFeatureEngineering( )

feature_names, storm_vars, environment_vars, object_properties_keys, _ = _feature_names_for_traditional_ml( )  
print("Number of features: ", len(feature_names))

get_time = personal_datetime( )

def calc_ensemble_stats( data ):
    '''
    Calculates the Ensemble Standard deviation, 10th percentile, 90th percentile, 
    mean, and 50th percentile
    '''
    # Calculate the ensemble statistics 
    ens_mean = np.nanmean( data, axis=1)
    #ens_median = np.percentile( data, 50, axis=1)
    #ens_90th = np.percentile( data, 90, axis=1)
    #ens_10th = np.percentile( data, 10, axis=1)
    ens_std = np.nanstd( data, axis=1, ddof=1)

    return ens_mean, ens_std #ens_10th, ens_median, ens_90th, ens_std

def concat_ensemble_data( strm_data, env_data):
    '''
    '''
    strm_ens_data = calc_ensemble_stats( strm_data ) 
    env_ens_data = calc_ensemble_stats( env_data ) 

    strm_ens_data = np.concatenate(strm_ens_data, axis=0)
    env_ens_data = np.concatenate(env_ens_data, axis=0)

    all_ens_data = np.concatenate((strm_ens_data, env_ens_data), axis = 0)

    return all_ens_data


def _load_environment_data(data_wofs, data_smryfiles, kwargs):
    '''
    Load environmental data 
    '''
    # Load the environmental ensemble data  
    env_data_smryfiles = data_smryfiles.load( variables=environment_vars[0], time_indexs=kwargs['time_indexs_env'], tag='ENV' )
    env_data_wofsdata  = data_wofs.load( variables=environment_vars[1], time_indexs=kwargs['time_indexs_env'] )
    env_data_i = np.concatenate(( env_data_smryfiles, env_data_wofsdata), axis = 1 )[0,:] # time is first axis 
    del env_data_smryfiles, env_data_wofsdata 

    return env_data_i

def _load_and_concat_radar_data( date, time, env_data_i ):
    #radar_data = [[ mrms.load_single_mrms_time(date_dir=date,
    #                              valid_datetime=get_time.determine_forecast_valid_datetime(date_dir = str(date), time_dir=time, fcst_time_idx=mrms_idx)[0],
    #                             var_name='DZ_CRESSMAN' ) for i in range(config.N_ENS_MEM) ] for mrms_idx in [0,3,6] ]
    #env_data = np.concatenate(( env_data_i, radar_data ), axis = 0 )
    #del radar_data

    return env_data_i

def _load_storm_data(data_wofs, data_smryfiles, kwargs):
    '''
    Load storm data 
    '''
    strm_data_smryfiles = data_smryfiles.load( variables=storm_vars[0], time_indexs=kwargs['time_indexs_strm'], tag='ENS' )
    strm_data_wofsdata  = data_wofs.load( variables=storm_vars[1], time_indexs=kwargs['time_indexs_strm'] )
    strm_data = np.concatenate(( strm_data_smryfiles, strm_data_wofsdata), axis = 1 )
    del strm_data_smryfiles, strm_data_wofsdata    

    strm_data_time_max = calc_time_max( strm_data )
    strm_data_time_mean = np.mean( strm_data, axis=0)
    strm_data_time_std = np.std( strm_data, axis = 0)
    all_strm_data = np.concatenate((strm_data_time_max, strm_data_time_mean, strm_data_time_std), axis=0)    

    return all_strm_data
    
def function_for_multiprocessing(date, time, kwargs ):
    '''
    A Function built exclusively for multiprocessing purposes.
    '''
    print ('\t Starting on {}-{}...'.format(date, time)) 
    # Load ensemble data  (examples, y, x, var)
    data_wofs = EnsembleData( date_dir =date, time_dir = time, base_path ='wofs_data')
    data_smryfiles = EnsembleData( date_dir =date, time_dir = time, base_path ='summary_files')

    # Load the environmental ensemble data  
    env_data_i = _load_environment_data(data_wofs, data_smryfiles, kwargs)    

    # Load MRMS Reflectivity 
    env_data = _load_and_concat_radar_data( date, time, env_data_i )

    # Load the storm ensemble data 
    all_strm_data = _load_storm_data(data_wofs, data_smryfiles, kwargs)

    # Concate storm and environment data
    all_ens_data = concat_ensemble_data( strm_data=all_strm_data, 
                                         env_data=env_data_i )

    in_path = join( config.OBJECT_SAVE_PATH, date )
    object_file = join(in_path, 'WOFS_%s_OBJECTS_%s-%s_%02d.nc' % (config.VARIABLE_ATTRIBUTES[variable_key]['title'], date, time, kwargs['fcst_time_idx']))
    ds = xr.open_dataset( object_file ) 
    ds_subset = ds[fn.object_properties_list] 
    df = ds_subset.to_dataframe()
    
    if extraction_method == 'hagelslag':
        forecast_objects = ds['Objects'].values
    ds.close( ) 
    del ds, ds_subset

    data = [ ] 
    ysu = [ ] 
    myj = [ ]
    mynn = [ ] 
    for mem_idx in range(config.N_ENS_MEM):
        df_at_mem_idx  = df.loc[ df['ensemble_member'] == mem_idx]
        x_object_cent = np.rint( df_at_mem_idx['obj_centroid_x'].values ).astype(int)
        y_object_cent = np.rint( df_at_mem_idx['obj_centroid_y'].values ).astype(int) 
        object_labels = df_at_mem_idx['label'].values
        good_idx = extract._remove_objects_near_boundary( x_object_cent, y_object_cent, NY=env_data.shape[-2], NX=env_data.shape[-1] )
        input_strm_data = all_strm_data[:, mem_idx, :,:]
        input_env_data = env_data[:, mem_idx,:,:]
        if extraction_method == 'inflow_sector':
            good_x_cent = x_object_cent[good_idx]     
            good_y_cent = y_object_cent[good_idx] 
            data_strm = extract._extract_storm_features_in_circle( input_strm_data, good_x_cent, good_y_cent )  
            data_env = extract._extract_environment_features_in_arcregion( input_env_data, good_x_cent, good_y_cent, avg_bunk_v_per_obj=data_strm[:,bunk_v_idx], avg_bunk_u_per_obj=data_strm[:,bunk_u_idx])
        elif extraction_method == 'hagelslag':
            good_object_labels = object_labels[good_idx]
            data_strm = extract._extract_features_from_object( input_strm_data, forecast_objects[mem_idx], good_object_labels )
            data_env = extract._extract_features_from_object( input_env_data, forecast_objects[mem_idx], good_object_labels, only_mean=True )
            data_ens = extract._extract_features_from_object( all_ens_data, forecast_objects[mem_idx], good_object_labels, only_mean=True )

        data_at_mem_idx = np.concatenate((data_strm, data_env, data_ens, df_at_mem_idx.values[good_idx, :]), axis = 1 )
        data.extend( data_at_mem_idx ) 
       
    data = np.array( data )
    if len(data) > 0:
        #Convert to dictionary with titles for the features
        initialization_time = [convert_to_seconds(time)]*np.shape(data)[0]
        initialization_time  = np.array( initialization_time )
        data = np.concatenate(( data, initialization_time[:, np.newaxis] ), axis = 1)

        full_data = {var: (['example'], data[:,i]) for i, var in enumerate(feature_names) }
        ds = xr.Dataset( full_data )
        fname = 'ML_WOFS_%s_%s-%s_%s_%s.nc' % (config.VARIABLE_ATTRIBUTES[variable_key]['title'], date, time, kwargs['fcst_time_idx'], extraction_method)
        out_path = join( config.ML_INPUT_PATH, str(date) )
        if debug:
            print(fname)
            ds.to_netcdf( fname )
        else:
            print('Writing {}...'.format(join(out_path, fname)))
            os.makedirs(os.path.dirname(join(out_path, fname)), exist_ok=True)
            ds.to_netcdf( path = join(out_path, fname) )
            ds.close( )
            del full_data


kwargs = {}
if debug:
    # 20170524-0000
    print ("DEBUG MODE") 
    fcst_time_idx = 0
    kwargs = {      
                  'time_indexs_strm': np.arange(config.N_TIME_IDX_FOR_HR+1),
                  'time_indexs_env': [ fcst_time_idx ],
                  'fcst_time_idx': fcst_time_idx}
    function_for_multiprocessing( date='20190524', time='0000', kwargs=kwargs )
else:
    datetimes = config.datetimes_ml
    for fcst_time_idx in [0, 6, 12]: #config.fcst_time_idx_set:
        print('\n Start Time:', datetime.now().time())
        print("Forecast Time Index: ", fcst_time_idx)
        kwargs = {'time_indexs_strm': np.arange(config.N_TIME_IDX_FOR_HR+1),
                  'time_indexs_env': [ fcst_time_idx ], 
                  'fcst_time_idx': fcst_time_idx}
        multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=8, func=function_for_multiprocessing, kwargs=kwargs)
        print('End Time: ', datetime.now().time())



