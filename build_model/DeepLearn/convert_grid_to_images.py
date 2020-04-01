from DeepLearn import DeepLearn, find_model_metafile, read_model_metadata, read_keras_model

import wofs.util.feature_names as fn
from wofs.data.loadEnsembleData import EnsembleData, calc_time_max
import numpy as np
from machine_learning.extraction.GridPointExtraction import PatchExtraction
from wofs.util import config
from itertools import product
from wofs.plotting.Plot import Plotting
from wofs.util.basic_functions import personal_datetime
from wofs.data.loadLSRs import loadLSR
from joblib import load

extract = PatchExtraction()
model = DeepLearn( )
get_time = personal_datetime()

VARIABLE_LIST = fn.env_vars_smryfiles + fn.env_vars_wofsdata + fn.storm_vars_smryfiles + fn.storm_vars_wofsdata
def convert_grid_to_images( date, time, kwargs ):
    '''
    '''
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
    ens_mean_data = np.mean( all_data, axis=1)

    x_rng = range(10, ens_mean_data.shape[-1]-10)    
    centers = product(x_rng,x_rng)

    examples = extract.extract_patch(ens_mean_data, centers)
    examples = np.array(examples)

    # reshaping
    new_examples = np.zeros((examples.shape[0], examples.shape[2], examples.shape[3], examples.shape[1]))
    for v in range(examples.shape[1]):
        new_examples[:,:,:,v] = examples[:,v,:,:]

    return {
            'predictor_names': VARIABLE_LIST,
            'predictor_matrix': np.array(new_examples),
            'shape': (len(x_rng),len(x_rng))    
            } 

date = '20180625'
time = '0000'
fcst_time_idx=6
kwargs = {'time_indexs_strm': np.arange( config.N_TIME_IDX_FOR_HR+1 ) + fcst_time_idx,
                  'time_indexs_env': [ fcst_time_idx ],
                  'fcst_time_idx': fcst_time_idx}

examples = convert_grid_to_images(date=date, time=time, kwargs=kwargs)

cnn_file_name = 'cnn_model.h5'

cnn_metafile_name = find_model_metafile(cnn_file_name)
cnn_metadata_dict = read_model_metadata('cnn_model_metadata.json')

forecast_probs = model.evaluate_cnn(
        cnn_model_object = read_keras_model(cnn_file_name),
        image_dict = examples,
        cnn_metadata_dict = cnn_metadata_dict
        )

calibrated_clf = load('iso_model_for_cnn.joblib')
forecast_probs = calibrated_clf.predict(forecast_probs)

forecast_probs_2D = np.reshape(forecast_probs, examples['shape'])
print (forecast_probs_2D.shape)

image = np.zeros((250,250))
image[10:250-10, 10:250-10] = forecast_probs_2D

kwargs = {'cblabel': 'Probability of LSR (within 15 km of a point)', 'alpha':0.7, 'extend': 'neither', 'cmap': 'wofs' }
plt = Plotting( date=date, z1_levels = np.arange(0.05, 1.05, 0.05), z2_levels = [0.], z3_levels=[0.], **kwargs )
fig, ax, map_ax, x, y = plt._create_fig( fig_num = 0, plot_map = True, figsize = (8, 9), sharey='row' )

valid_date_and_time, initial_date_and_time = get_time.determine_forecast_valid_datetime( date_dir=str(date), time_dir=time, fcst_time_idx=fcst_time_idx )
load_lsr = loadLSR(date_dir=date, date=valid_date_and_time[0], time=valid_date_and_time[1])
hail_ll = load_lsr.load_hail_reports( )
torn_ll = load_lsr.load_tornado_reports( )
wind_ll = load_lsr.load_wind_reports( )
lsr_points = {'hail': hail_ll, 'tornado': torn_ll, 'wind': wind_ll }

contours = plt.spatial_plotting(fig, ax, x, y, z1=image, map_ax=map_ax[0], lsr_points=lsr_points)
plt._add_major_colobar(fig, contours, label='Probability of Severe Weather',fontsize=18, coords=[0.92, 0.41, 0.03, 0.1])
fname = f'individual_fcst_probs_CNN_{date}_{time}_fcst_time_idx={fcst_time_idx}.png'
plt._save_fig( fig=fig, fname = fname)


