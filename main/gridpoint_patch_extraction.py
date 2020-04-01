import numpy as np
import itertools
import xarray as xr 
from os.path import join, exists
import os 
from datetime import datetime

from wofs.data.loadEnsembleData import calc_time_max, calc_time_tendency, EnsembleData
from wofs.data.loadMRMSData import MRMSData
from wofs.util import config 
from machine_learning.extraction.GridPointExtraction import PatchExtraction
from wofs.util.MultiProcessing import multiprocessing_per_date
from wofs.util.feature_names import _feature_names_for_traditional_ml
import wofs.util.feature_names as fn 
from wofs.util.basic_functions import personal_datetime

""" usage: stdbuf -oL python -u  gridpoint_patch_extraction.py 2 > & log_deep_learn_patch_extraction & """
debug = False
variable_key = 'updraft' 
var = 'LSR_15km_grid'

extract = PatchExtraction()
mrms = MRMSData()
get_time = personal_datetime()
VARIABLE_LIST = fn.env_vars_smryfiles + fn.env_vars_wofsdata + fn.storm_vars_smryfiles + fn.storm_vars_wofsdata

def function_for_multiprocessing( date, time, kwargs ):
    '''
    Function for multiprocessing.
    '''
    print(date,time)
    data_wofs = EnsembleData( date_dir =date, time_dir = time, base_path ='wofs_data')
    data_smryfiles = EnsembleData( date_dir =date, time_dir = time, base_path ='summary_files')

    inflow_env_data1 = data_smryfiles.load( variables=fn.env_vars_smryfiles, time_indexs=kwargs['time_indexs_env'], tag='ENV' )
    inflow_env_data2 = data_wofs.load( variables=fn.env_vars_wofsdata, time_indexs=kwargs['time_indexs_env'] )
    inflow_env_data_combined = np.concatenate(( inflow_env_data1, inflow_env_data2), axis = 1 )[0,:] # time is first axis 
    del inflow_env_data1, inflow_env_data2

    intra_strm_data1 = data_smryfiles.load( variables=fn.storm_vars_smryfiles, time_indexs=kwargs['time_indexs_strm'], tag='ENS' )
    intra_strm_data2 = data_wofs.load( variables=fn.storm_vars_wofsdata, time_indexs=kwargs['time_indexs_strm'] )
    intra_strm_data = np.concatenate(( intra_strm_data1, intra_strm_data2), axis = 1 )
    intra_strm_data_time_max = calc_time_max( intra_strm_data )
    del intra_strm_data1, intra_strm_data2, intra_strm_data 

    all_data = np.concatenate(( inflow_env_data_combined, intra_strm_data_time_max ), axis =0 )

    ens_mean_strm_data = np.percentile( intra_strm_data_time_max, 90, axis=0)  
    ens_mean_data = np.mean( all_data, axis=1)

    ens_data = np.concatenate(( ens_mean_strm_data, ens_mean_data), axis=0)

    print (ens_data.shape)

    valid_date_and_time, initial_date_and_time = get_time.determine_forecast_valid_datetime( date_dir=str(date), time_dir=time, fcst_time_idx=fcst_time_idx )
    lsrs = mrms.load_gridded_lsrs(var=var, date_dir=str(date), valid_datetime=valid_date_and_time)

    centers, patch_labels = extract.subsample( lsrs )
    examples = extract.extract_patch(ens_data, centers)

    examples = np.array(examples)

    if len(examples) > 0:
        data = {var: (['example', 'y', 'x'], examples[:,i,:,:]) for i, var in enumerate(VARIABLE_LIST ) }
        data['label'] = (['example'], patch_labels)
        ds = xr.Dataset( data )
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}
        out_path = join( config.ML_INPUT_PATH, str(date) )
        fname = 'DL_WOFS_%s-%s_%s_generic_patches.nc' % (date, time, fcst_time_idx)    
        if debug:
            ds.to_netcdf(path=fname, encoding=encoding)
        else:
            print('Writing {}...'.format( join(out_path,fname) ))
            ds.to_netcdf(path = join(out_path,fname), encoding=encoding) 
        ds.close( ) 
        del ds, data 


kwargs = { }
if debug:
    print("DEBUG MODE")
    fcst_time_idx = 0 
    kwargs = {'time_indexs_strm': np.arange( config.N_TIME_IDX_FOR_HR+1 ) + fcst_time_idx,
                  'time_indexs_env': [ fcst_time_idx ],
                  'fcst_time_idx': fcst_time_idx} 
    function_for_multiprocessing( date='20180501', time='2300', kwargs=kwargs )
else:
    datetimes = config.datetimes_ml
    for fcst_time_idx in [6]: #config.fcst_time_idx_set:
        print('\n Start Time:', datetime.now().time())
        print("Forecast Time Index: ", fcst_time_idx)
        kwargs = {'time_indexs_strm': np.arange( config.N_TIME_IDX_FOR_HR+1 ) + fcst_time_idx,
                  'time_indexs_env': [ fcst_time_idx ],
                  'fcst_time_idx': fcst_time_idx}
        multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=8, func=function_for_multiprocessing, kwargs=kwargs)
        print('End Time: ', datetime.now().time())


