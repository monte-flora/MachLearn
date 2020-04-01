import sys
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/wofs/plotting')
sys.path.append('/home/monte.flora/wofs/data')
sys.path.append('/home/monte.flora/wofs/processing')
sys.path.append('/home/monte.flora/hagelslag/hagelslag/processing')
import config
from Plot import Plotting
import xarray as xr
from loadMRMSData import MRMSData
from os.path import join
from basic_functions import personal_datetime
import numpy as np 
from EnhancedWatershedSegmenter import EnhancedWatershed
from ObjectIdentification import ObjectIdentification, QualityControl
from scipy.ndimage import gaussian_filter, maximum_filter 

# Days with misses: 20170522(fold=0), 20170519, 20170511, 20180512
# 20180501(fold=0), 20170516(fold=4,6), 
# Looks like if misses occur, it is for the whole day! 
# XGBoost 20170517 0000 8
date = '20180501'
time = '0000'
fcst_time_idx = 6
fold = 0 
verify_var = 'AZSHR'
var_mrms = 'LOW_CRESSMAN'
var_newse = 'uh_0to2'
max_nghbrd = 0 
get_time = personal_datetime( )
mrms = MRMSData( )
name = 'Rotation Objects (QC9)'

# Parameters of the CNN
num_conv_blocks=2
num_dense_layers=2
num_conv_layers_in_a_block=1
first_num_filters = 8
use_batch_normalization = True
kernel_size = 3
dropout_fraction = 0.5
l1_weight = 0.
l2_weight = 0.001
activation_function_name = 'leaky_relu'
pooling_type = 'max'
dim_option = '2D'
conv_type = 'separable'
patience = 8
min_delta = 1e-5

valid_date_and_time, initial_date_and_time = get_time.determine_forecast_valid_datetime( date_dir=str(date), time_dir=time, fcst_time_idx=fcst_time_idx )
in_fcst_path = join( config.ML_FCST_PATH, str(date) )
RF_fname = join( in_fcst_path,'%s_fcst_verify_var=%s_%s-%s_%s_fold=%s.nc' % ( 'RandomForest', verify_var, date, time, fcst_time_idx, fold ))
XGB_fname = join( in_fcst_path,'%s_fcst_verify_var=%s_%s-%s_%s_fold=%s.nc' % ( 'XGBoost', verify_var, date, time, fcst_time_idx, fold ))
CNN_fname = join( in_fcst_path,'%s_fcst_verify_var=%s_%s-%s_%s_fold=%s.nc' % ( 'CNN', verify_var, date, time, fcst_time_idx, fold ))

in_fcst_path = join( config.WOFS_PROBS_PATH, str(date) )
wofs_fname = join(in_fcst_path, 'WOFS_%s_PROBS_%s-%s_%02d.nc' % (config.variable_attrs['low-level']['title'], date, time, fcst_time_idx))

ds_RF = xr.open_dataset( RF_fname )
ds_XGB = xr.open_dataset( XGB_fname )
ds_CNN = xr.open_dataset( CNN_fname )
ds_wofs = xr.open_dataset( wofs_fname )

RF_probs = ds_RF['Ens. Avg. Probability'].values
XGB_probs = ds_XGB['Ens. Avg. Probability'].values
CNN_probs = ds_CNN['Ens. Avg. Probability'].values
wofs_probs = ds_wofs['Ensemble Probability (QC3)'].values

# NMEP OF CNN PROBS
CNN_all_probs = ds_CNN['Probability'].values
nmep = np.zeros(( CNN_all_probs.shape ))
for i in range(CNN_all_probs.shape[0]):
    nmep[i,:,:] = maximum_filter( CNN_all_probs[i,:,:], 3)
combined_probs = np.mean(nmep, axis = 0 )
combined_probs[CNN_probs==0] = 0 

#combined_probs = np.array([RF_probs, XGB_probs, CNN_probs])
#combined_probs = np.amax( combined_probs, axis = 0 ) 

obj_id = ObjectIdentification( )
qc = QualityControl( )
processed_data = np.round(1000.*combined_probs ,0).astype(int)
#processed_data = gaussian_filter( np.round(1000.*combined_probs ,0), sigma = 1 ).astype(int)
watershed_params_fcst = {'min_thresh': 1,
                         'max_thresh': 750,
                         'data_increment': 1,
                         'delta': 1000,
                         'size_threshold_pixels': 200,
                         'local_max_area': 16 }
labels_fcst, props_fcst = obj_id.label( processed_data, method='watershed', **watershed_params_fcst)
qc_params = {'min_area':10.}
labels_fcst_combined, qc_object_props = qc.quality_control(object_labels=labels_fcst, object_properties=props_fcst, input_data = combined_probs, qc_params=qc_params )

labels_fcst_wofs = ds_wofs['Probability Objects (QC3)'].values
labels_fcst_RF = ds_RF['Probability Objects'].values
labels_fcst_XGB = ds_XGB['Probability Objects'].values
labels_fcst_CNN = ds_CNN['Probability Objects'].values

labels_obs, az_shear = mrms.load_az_shr_tracks( var=config.variable_attrs['low-level']['var_mrms'], name=name, date_dir=str(date), valid_datetime=valid_date_and_time )

kwargs = {'cblabel': 'Probability of Low-level Rotation', 'alpha':0.7, 'extend': 'neither', 'cmap': 'wofs' }
plt = Plotting( date=date, z1_levels = np.arange(0.0001, 1.05, 0.05), z2_levels = [0.], z3_levels=[0.], **kwargs ) 
fig, axes, map_axes, x, y = plt._create_fig( fig_num = 0, plot_map = True, sub_plots = (1, 5), figsize = (16, 18), sharey='row' )

probs = [ RF_probs, XGB_probs, CNN_probs, combined_probs, wofs_probs ] 
prob_objects = [ labels_fcst_RF, labels_fcst_XGB, labels_fcst_CNN, labels_fcst_combined, labels_fcst_wofs] 
titles = [ 'RF', 'XGB', 'CNN', 'CNN NMEP', 'WOFS' ]
for i, ax in enumerate(fig.axes):    
    contours = plt.spatial_plotting( fig=fig, ax=ax, map_ax = map_axes[i], x=x, y=y, z1=probs[i], z2=labels_obs, z3=prob_objects[i], title=titles[i] )

plt._add_major_colobar(fig, contours, label='Probability of Low-level Rotation',fontsize=18, coords=[0.92, 0.41, 0.03, 0.1])
fname = 'model_comparsion_fcst_probs_%s_%s.png' % (date, time) 
plt._save_fig( fig=fig, fname = fname)

