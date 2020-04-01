import sys
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/wofs/plotting')
sys.path.append('/home/monte.flora/wofs/data')
sys.path.append('/home/monte.flora/wofs/processing')
import config
from Plot import Plotting
import xarray as xr
from loadMRMSData import MRMSData
from os.path import join
from basic_functions import personal_datetime
import numpy as np 
from EnhancedWatershedSegmenter import EnhancedWatershed
from ObjectIdentification import ObjectIdentification, QualityControl
from scipy.ndimage import maximum_filter, gaussian_filter

# Days with misses: 20170522(fold=1,2), 20170519, 20170511, 20180512
# Looks like if misses occur, it is for the whole day! 
date = '20180601'
time = '0200'
variable_key = 'low-level' 
model_name = 'WOFS'
fcst_time_idx = 0
fold = 0 
max_nghbrd = 0
var_mrms = 'LOW_CRESSMAN'
get_time = personal_datetime( )
mrms = MRMSData( )
qc = QualityControl( )
name = 'Rotation Objects (QC8)'

min_thresh = 2
max_thresh = 75
data_increment = 10
delta = 7
area = 200 
local_max_area = 16 
max_filter_size = 2
sigma = 1 
watershed_params_fcst = {'min_thresh': min_thresh, 
                         'max_thresh':max_thresh, 
                         'data_increment': data_increment, 
                         'delta': delta, 
                         'size_threshold_pixels': area }

watershed = EnhancedWatershed( min_thresh=min_thresh, data_increment=data_increment, max_thresh=max_thresh, delta=delta, size_threshold_pixels=area, local_max_area=local_max_area)  
valid_date_and_time, initial_date_and_time = get_time.determine_forecast_valid_datetime( date_dir=str(date), time_dir=time, fcst_time_idx=fcst_time_idx )
print valid_date_and_time
if model_name == 'RandomForest':
    in_path = join( config.fcst_dir_object_based, str(date) )
    fname = join( in_path, '%s_fcst_%s-%s_%s_fold=%s' % ( model_name, date, time, fcst_time_idx, fold ))
    ds = xr.open_dataset( fname )
    probs = ds['Ens. Avg. Probability'].values
    original_labels = np.zeros(( probs.shape ))
        
elif model_name == 'WOFS':
    in_fcst_path = join( config.forecast_prob_dir, str(date) )
    fcst_fname = join(in_fcst_path, 'WOFS_%s_PROBS_%s-%s_%02d.nc' % (config.variable_attrs[variable_key]['title'], date, time, fcst_time_idx))
    ds = xr.open_dataset( fcst_fname )
    probs = ds['Ensemble Probability'].values
    #original_labels = ds['Probability Objects'].values

obj_id = ObjectIdentification( )
processed_data = gaussian_filter( np.round(100.*probs,0), sigma = 1 ) 
labels_fcst, props_fcst = obj_id.label( processed_data, method='watershed', **watershed_params_fcst)
pixels, quant_values = watershed.quantize( processed_data )
marked, original_marked = watershed.find_local_maxima( processed_data )

qc_params = {'min_area':10.}  
qc_object_labels, qc_object_props = qc.quality_control(object_labels=labels_fcst, object_properties=props_fcst, input_data = probs, **qc_params )

for region in qc_object_props:
    print "Label {} has area of {}".format(region.label, round(region.area,1))

data = { }
data['Probability'] = (['Y', 'X'], probs)
data['Objects'] = (['Y', 'X'], qc_object_labels)
data['Processed'] = (['Y', 'X'], processed_data)
data['Quantize'] =  (['Y', 'X'], quant_values)
data['Final Marked'] =  (['Y', 'X'], marked)
data['Original Marked'] =  (['Y', 'X'], original_marked)

ds = xr.Dataset( data )
nc_fname = '%s_fcst_probs_objects_%s_%s_min=%s_max=%s_incr=%s_delt=%s_area=%s_maxfilt=%s_gauss=%s.nc' % (model_name, date, time, min_thresh, max_thresh, data_increment, delta, area, max_filter_size, sigma)
ds.to_netcdf( nc_fname )

labels_obs, _ = mrms.load_az_shr_tracks( var=var_mrms, name=name, date_dir=str(date), valid_datetime=valid_date_and_time)

kwargs = {'cblabel': 'Probability of Low-level Rotation', 'alpha':0.7, 'extend': 'neither', 'cmap': 'wofs' }
plt = Plotting( date=date, z1_levels = np.arange(0.001, 1.05, 0.05), z2_levels = [0.], z3_levels=[0.], **kwargs ) 
fig, axes, map_axes, x, y = plt._create_fig( fig_num = 0, plot_map = True, figsize = (8, 9))

plt.spatial_plotting( fig=fig, ax=axes, map_ax = map_axes[0], x=x, y=y, z1=probs, z3=qc_object_labels, z4=labels_obs, title='' )

fname = '%s_fcst_probs_objects_%s_%s_min=%s_max=%s_incr=%s_delt=%s_area=%s_maxfilt=%s_gauss=%s.png' % (model_name, date, time, min_thresh, max_thresh, data_increment, delta, area, max_filter_size, sigma) 
plt._save_fig( fig=fig, fname = fname)

