import os 
from os.path import join 
from wofs.util import config
from wofs.plotting.Plot import Plotting
import xarray as xr
from wofs.data.loadMRMSData import MRMSData
from wofs.util.basic_functions import personal_datetime
import numpy as np 
from wofs.data.loadLSRs import loadLSR
from wofs.data.loadWWAs import loadWWA
from wofs.processing.ObjectIdentification import label, QualityControl

# Days with misses: 20170522(fold=0), 20170519, 20170511, 20180512
# 20180501(fold=0), 20170516(fold=4,6), 
# Looks like if misses occur, it is for the whole day! 
# XGBoost 20170517 0000 8
model_name = 'XGBoost'
date = '20180502'
time = '2300'

variable_key = 'low-level'

fcst_time_idx = 12
verify_var = 'matched_to_LSRs_30km'
calibrate=False
get_time = personal_datetime( )
mrms = MRMSData( )

in_fcst_path = join( config.ML_FCST_PATH, str(date) )
f = [x for x in os.listdir(in_fcst_path) if 'calibrate' in x][0]
fold = f[-4] 

def load_probs(model_name):
    fname = join( in_fcst_path, f'{model_name}_fcst_calibrate={calibrate}_verify_var={verify_var}_{date}-{time}_{fcst_time_idx}_fold={fold}.nc')
    ds = xr.open_dataset( fname )
    probs = ds['2D Probabilities'].values
    return probs

rf_probs = load_probs('RandomForest')
xgb_probs = load_probs('XGBoost')

in_fcst_path = join( config.WOFS_PROBS_PATH, str(date) )
fcst_fname = join(in_fcst_path, 'WOFS_%s_PROBS_%s-%s_%02d.nc' % (config.VARIABLE_ATTRIBUTES[variable_key]['title'], date, time, fcst_time_idx))

ds_fcst = xr.open_dataset( fcst_fname )
prob_name = 'Ensemble Probability (QC4)'
prob_object_name = 'Probability Objects (QC4)'
ens_probs = ds_fcst[prob_name].values
labels_fcst = ds_fcst[prob_object_name].values

ens_probs_copy = np.zeros(ens_probs.shape)
for label in np.unique(labels_fcst):
    ens_probs_copy[labels_fcst==label] = np.amax( ens_probs[labels_fcst==label] )


valid_date_and_time, initial_date_and_time = get_time.determine_forecast_valid_datetime( date_dir=str(date), time_dir=time, fcst_time_idx=fcst_time_idx )
load_lsr = loadLSR(date_dir=date, date=valid_date_and_time[0], time=valid_date_and_time[1])
hail_ll = load_lsr.load_hail_reports( )
torn_ll = load_lsr.load_tornado_reports( )
wind_ll = load_lsr.load_wind_reports( )
lsr_points = {'hail': hail_ll, 'tornado': torn_ll, 'wind': wind_ll }

kwargs = {'cblabel': 'Probability of Low-level Rotation', 'alpha':0.7, 'extend': 'neither', 'cmap': 'wofs' }
plt = Plotting( date=date, z1_levels = np.arange(0.05, 1.05, 0.05), z2_levels = [0.], z3_levels=[0.], **kwargs ) 
fig, axes, map_axes, x, y = plt._create_fig( fig_num = 0, sub_plots =(2,2), plot_map = True, figsize = (8, 9), sharey='row' )

contours = plt.spatial_plotting(fig, axes.flat[0], x, y, z1=np.ma.masked_where(ens_probs_copy<0.0001, ens_probs_copy), map_ax=map_axes[0], lsr_points=lsr_points, z1_is_integers=True )
plt.spatial_plotting(fig, axes.flat[1], x, y, z1= np.ma.masked_where( rf_probs == 0, rf_probs ),  map_ax=map_axes[1], lsr_points=lsr_points, z1_is_integers=True)
plt.spatial_plotting(fig, axes.flat[2], x, y, z1= np.ma.masked_where( xgb_probs == 0, xgb_probs ), map_ax=map_axes[2], lsr_points=lsr_points, z1_is_integers=True)

plt._add_major_colobar(fig, contours, label='Probability of Severe Weather',fontsize=18, coords=[0.92, 0.41, 0.03, 0.1])
fname = f'individual_fcst_probs_{model_name}_{date}_{time}_fcst_time_idx={fcst_time_idx}.png'
plt._save_fig( fig=fig, fname = fname)

