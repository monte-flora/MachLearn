import numpy as np
import itertools
import xarray as xr 
from os.path import join, exists
import os 
import sys
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

#personal modules 
from wofs.data.loadEnsembleData import EnsembleData
from wofs.data.loadEnsembleData import calc_time_max, calc_time_tendency
#from wofs.data.loadMRMSData import MRMSData
from wofs.util import config 
from machine_learning.extraction.StormBasedFeatureEngineering import StormBasedFeatureEngineering
#from wofs.util.MultiProcessing import multiprocessing_apply_async
from wofs.util.feature_names import _feature_names_for_traditional_ml
import wofs.util.feature_names as fn 
from wofs.util.basic_functions import convert_to_seconds, personal_datetime
get_time = personal_datetime( )
extract = StormBasedFeatureEngineering( )

""" usage: stdbuf -oL python -u ensemble_feature_extraction.py  2 > & log_extract & """
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
    datasets = datasets.load()

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

    datasets_time_stats = xr.merge( (
                                      time_max_dataset, 
                                      time_std_dataset
                                    )
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


def _save_netcdf( data, date, time, fcst_time_idx ):
    '''
    saves a netcdf file
    '''
    ds = xr.Dataset( data )
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars} 
    fname = join(config.ML_INPUT_PATH, str(date), f'PROBABILITY_OBJECTS_{date}-{time}_{fcst_time_idx:02d}.nc')
    ###fname = f'PROBABILITY_OBJECTS_{date}-{time}_{fcst_time_idx:02d}.nc'
    print( f"Writing {fname}...")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    ds.to_netcdf( path = fname, encoding=encoding )
    ds.close( )
    del ds, data 
 
##############################
#      MAIN FUNCTION         #
##############################

def worker(date,time,fcst_time_idx):
    """
    worker function for multiprocessing
    """
    fname = join(config.ML_INPUT_PATH, str(date), f'PROBABILITY_OBJECTS_{date}-{time}_{fcst_time_idx:02d}.nc')
    duration=6
    time_indexs = np.arange(duration+1)+fcst_time_idx
    print ('\t Starting on {}-{}-{}...'.format(date, time, fcst_time_idx))
    # Read in the probability object file 
    object_file = join(
                    config.OBJECT_SAVE_PATH,
                    date,
                    f'updraft_ensemble_objects_{date}-{time}_t:{fcst_time_idx}.nc'
                  )
    
    if not exists(object_file):
        raise Exception(f'{object_file} does not exist! This is expected since we are using too many times')

    if exists(fname):
        raise Exception(f'{fname} already exists!') 
    
    object_dataset = xr.open_dataset( object_file )

    forecast_objects = object_dataset['Probability Objects'].values
    try:
        ds_subset = object_dataset[fn.probability_object_props]
    except:
        raise Exception(f'KeyError for {date}-{time}-{fcst_time_idx}; No Objects in domain!')

    df = ds_subset.to_dataframe()
    object_labels = df['label']
 
    ds = load_dataset( 
                   smryfile_variables = fn.smryfile_variables,
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
    combined_dataset = xr.merge( 
                             (storm_dataset,
                              environment_dataset)
                           )

    # Calculate the ensemble statistics 
    ensemble_dataset = calc_ensemble_stats( datasets=combined_dataset )

    data = [ ]
    strm_array =  {var: storm_dataset[var].values[:, :,:] for var in list(storm_dataset.data_vars)}
    ensemble_array = {var: ensemble_dataset[var].values[:,:] for var in list(ensemble_dataset.data_vars)}
    
    ensemble_dataset.close()
    storm_dataset.close()
    ds.close()
    del ensemble_dataset, storm_dataset, ds

    # Extract the features 
    data_ens, ens_feature_names = extract.extract_spatial_features_from_object( ensemble_array, forecast_objects, object_labels, only_mean=True, mem_idx=None )
    data_ens_amp, ens_amp_feature_names = extract.extract_amplitude_features_from_object( strm_array, forecast_objects, object_labels)
    data = np.concatenate((data_ens, data_ens_amp, df.values), axis = 1 )
       
    data = np.array( data )
    if len(data) > 0:
        #Convert to dictionary with titles for the features
        initialization_time = [convert_to_seconds(time)]*np.shape(data)[0]
        initialization_time  = np.array( initialization_time )
        data = np.concatenate(( data, initialization_time[:, np.newaxis] ), axis = 1)
        feature_names = ens_feature_names + ens_amp_feature_names + fn.probability_object_props + fn.additional_vars
        full_data = {var: (['example'], data[:,i]) for i, var in enumerate(feature_names) }
        del data 
        _save_netcdf( data=full_data, date=date, time=time, fcst_time_idx=fcst_time_idx )

# /work/mflora/ML_DATA/INPUT_DATA/20170508/PROBABILITY_OBJECTS_20170508-0130_06.nc
#worker(date='20190507', time='4330', fcst_time_idx=6)


