import numpy as np
import itertools
import xarray as xr 
from os.path import join, exists
import os 
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

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
get_time = personal_datetime( )
extract = StormBasedFeatureEngineering( )

""" usage: stdbuf -oL python -u ml_feature_extraction.py 2 > & log_extract & """
debug = False

def load_dataset( smryfile_variables, wofsdata_variables, date, time, time_indexs):
    '''
    Loads data from Pat's summary files and the summary files
    I generated.
    ---------------
    Args:
        data, string of date directory (format: YYMMDD)
        time, string of time directory (format: HHmm )
        smryfile_variables, dictionary with keys of 'ENS' and 'ENV' with 
                            lists of the summary files variables to load
        wofsdata_variables, list of wofs data variables to load 
        time_indexs,

    Return:
        datasets, xarray dataset
    '''
    # Load the summary file data 
    datasets = [ ]
    for tag in ['ENV', 'ENS']:
        instance = EnsembleData( date_dir=date, 
                                 time_dir=time, 
                                 base_path ='summary_files')
        ds = instance.load( variables = smryfile_variables[tag],
                            time_indexs = time_indexs, 
                            tag = tag )           
        datasets.append( ds )

    instance = EnsembleData( date_dir=date, 
                             time_dir=time, 
                             base_path ='wofs_data')

    ds = instance.load( variables = wofsdata_variables,
                        time_indexs = time_indexs,
                       )
    datasets.append( ds )
    datasets = xr.merge( datasets )
    # Ens. Mem., Y, X)
    datasets = datasets.rename( {'Ens. Mem.': 'NE', 'Y':'NY', 'X':'NX'} )

    # Only keep the initial time for the environmental variables
    for var in fn.environmental_variables:
        datasets[var] = (['NE', 'NY', 'NX'], datasets[var].values[0,:])

    datasets = datasets.rename( {'10-m_bulk_shear': '10-500m_bulk_shear'} )

    return datasets


def calc_temporal_stats( datasets ):
    '''
    Calculate the time-max, mean, and standard deviation
    of the intra-storm variables.
    -----------------
    Args: 
        datasets, an xarray dataset
    Returns:
        datasets_time_stats
    '''
    datasets = datasets[fn.storm_variables]
    # Calculate the time-mean value for the intra-storm variables
    time_mean_dataset = datasets.mean( dim='time')
    time_mean_dataset = time_mean_dataset.rename( 
               {var:var+'_time_mean' for var in list(datasets.data_vars)}
               )
    # Calculate the time-max value for the intra-storm variables
    time_max_dataset = datasets.max( dim='time')
    time_max_dataset = time_max_dataset.rename( 
               {var:var+'_time_max' for var in list(datasets.data_vars)}
               ) 
     # Calculate the time-std value for the intra-storm variables
    time_std_dataset = datasets.std( dim='time')
    time_std_dataset = time_std_dataset.rename(   
               {var:var+'_time_std' for var in list(datasets.data_vars)}
               )

    datasets_time_stats = xr.merge( (time_mean_dataset,
                                      time_max_dataset, 
                                      time_std_dataset)
                                    )
    return datasets_time_stats


def calc_ensemble_stats( datasets ):
    '''
    Calculates the ensemble mean and standard deviation.
    Performed on both the environmental and storm variables.
    ------------------
    Args:
        datasets, xarray dataset
    Returns:
        
    '''
    # Calculate the ensemble mean
    ensemble_mean_dataset = datasets.mean( dim ='NE' )
    ensemble_mean_dataset = ensemble_mean_dataset.rename( 
            {var:var+'_ens_mean' for var in list(datasets.data_vars)}
                        )
    # Calculate the ensemble standard deviation 
    ensemble_std_dataset = datasets.std( dim ='NE' )
    ensemble_std_dataset = ensemble_std_dataset.rename(
            {var:var+'_ens_std' for var in list(datasets.data_vars)}
                        )
    
    ensemble_dataset = xr.merge( (ensemble_mean_dataset,
                                  ensemble_std_dataset )
                                )

    return ensemble_dataset

def function_for_multiprocessing(date, time, kwargs ):
    '''
    A Function built exclusively for multiprocessing purposes.
    '''
    extraction_method = 'hagelslag'
    fcst_time_idx = kwargs['fcst_time_idx']
    time_indexs = kwargs['time_indexs']
    print ('\t Starting on {}-{}...'.format(date, time)) 
    ds = load_dataset( smryfile_variables = fn.smryfile_variables,
                   wofsdata_variables = fn.wofsdata_variables,
                   time_indexs = time_indexs,
                   date = date,
                   time = time
                  )

    dims = dict(ds.sizes)
    # Calculate the temporal statistics 
    storm_dataset = calc_temporal_stats( datasets=ds )
    # Combine the environmental and intra-storm variables back together
    environment_dataset = ds[fn.environmental_variables]
    combined_dataset = xr.merge( (storm_dataset,
                                  environment_dataset)
                                )
    # Calculate the ensemble statistics 
    ensemble_dataset = calc_ensemble_stats( datasets=combined_dataset )

    in_path = join( config.OBJECT_SAVE_PATH, date )
    object_file = join(in_path, f'WOFS_UPDRAFT_SWATH_OBJECTS_{date}-{time}_{fcst_time_idx:02d}.nc')
    ds = xr.open_dataset( object_file ) 
    ds_subset = ds[fn.object_properties_list] 
    df = ds_subset.to_dataframe()
    
    forecast_objects = ds['Objects'].values
    data = [ ]

    strm_array =  {var: storm_dataset[var].values[:, :,:] for var in list(storm_dataset.data_vars)}
    environment_array = {var: environment_dataset[var].values[:,:] for var in list(environment_dataset.data_vars)}
    ensemble_array = {var: ensemble_dataset[var].values[:,:] for var in list(ensemble_dataset.data_vars)}

    storm_dataset.close()
    environment_dataset.close()
    ensemble_dataset.close()

    for mem_idx in range(config.N_ENS_MEM):
        df_at_mem_idx  = df.loc[ df['ensemble_member'] == mem_idx]
        x_object_cent = np.rint( df_at_mem_idx['obj_centroid_x'].values ).astype(int)
        y_object_cent = np.rint( df_at_mem_idx['obj_centroid_y'].values ).astype(int) 
        object_labels = df_at_mem_idx['label'].values
        good_idx = extract._remove_objects_near_boundary( x_object_cent, 
                                                          y_object_cent, 
                                                          NY=dims['NY'], 
                                                          NX=dims['NX'] )
        # ['NE', 'NX', 'NY', 'time']
        good_object_labels = object_labels[good_idx]
        data_strm, strm_feature_names = extract.extract_spatial_features_from_object( strm_array, forecast_objects[mem_idx], good_object_labels, mem_idx=mem_idx )
        data_env, env_feature_names = extract.extract_spatial_features_from_object( environment_array, forecast_objects[mem_idx], good_object_labels, only_mean=True, mem_idx=mem_idx )
        data_ens, ens_feature_names = extract.extract_spatial_features_from_object( ensemble_array, forecast_objects[mem_idx], good_object_labels, only_mean=True, mem_idx=None )
        data_ens_amp, ens_amp_feature_names = extract.extract_amplitude_features_from_object( strm_array, forecast_objects[mem_idx], good_object_labels)

        data_at_mem_idx = np.concatenate((data_strm, data_env, data_ens, data_ens_amp, df_at_mem_idx.values[good_idx, :]), axis = 1 )
        data.extend( data_at_mem_idx ) 
       
    data = np.array( data )
    if len(data) > 0:
        #Convert to dictionary with titles for the features
        initialization_time = [convert_to_seconds(time)]*np.shape(data)[0]
        initialization_time  = np.array( initialization_time )
        data = np.concatenate(( data, initialization_time[:, np.newaxis] ), axis = 1)
        feature_names = strm_feature_names + env_feature_names + ens_feature_names + ens_amp_feature_names + fn.object_properties_list + fn.additional_vars
        full_data = {var: (['example'], data[:,i]) for i, var in enumerate(feature_names) }
        ds = xr.Dataset( full_data )
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars} 

        if debug:
            print ('Saving file...') 
            ds.to_netcdf( f'ML_WOFS_UPDRAFT_SWATH_{date}-{time}_{fcst_time_idx}.nc', encoding=encoding )
        else:
            fname = join(config.ML_INPUT_PATH, str(date), f'ML_WOFS_UPDRAFT_SWATH_{date}-{time}_{fcst_time_idx}.nc')
            print( f"Writing {fname}...")
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            ds.to_netcdf( path = fname, encoding=encoding )
            ds.close( )
            del full_data


kwargs = {}
if debug:
    # 20170524-0000
    print ("DEBUG MODE") 
    fcst_time_idx = 18
    kwargs = {      
                  'time_indexs': np.arange(config.N_TIME_IDX_FOR_HR+1)+fcst_time_idx,
                  'fcst_time_idx': fcst_time_idx}
    function_for_multiprocessing( date='20190524', time='0000', kwargs=kwargs )
else:
    datetimes = config.datetimes_ml
    for fcst_time_idx in config.fcst_time_idx_set:
        print('\n Start Time:', datetime.now().time())
        print("Forecast Time Index: ", fcst_time_idx)
        kwargs = {'time_indexs': np.arange(config.N_TIME_IDX_FOR_HR+1)+fcst_time_idx,
                  'fcst_time_idx': fcst_time_idx}
        multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=8, func=function_for_multiprocessing, kwargs=kwargs)
        print('End Time: ', datetime.now().time())

