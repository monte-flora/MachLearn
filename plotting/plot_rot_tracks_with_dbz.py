import sys
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/wofs/plotting')
sys.path.append('/home/monte.flora/wofs/data')
sys.path.append('/home/monte.flora/wofs/processing')
import config
from Plot import Plotting
import xarray as xr
from loadMRMSData import MRMSData
from loadEnsembleData import EnsembleData
from os.path import join
from basic_functions import personal_datetime, get_key
import numpy as np 
from ObjectMatching import ObjectMatching
from ObjectIdentification import ObjectIdentification
from skimage.measure import regionprops

################################################################################
''' Plots hour-long rotation tracks with reflectivity valid at a single time '''
#################################################################################

# Days with misses: 20170522, 20170519, 20170511
# Looks like if misses occur, it is for the whole day! 

date = '20170516'
time = '2300'
fcst_time_idx = 0
var_mrms = 'LOW_CRESSMAN'
var_newse = 'uh_0to2'
max_nghbrd = 0 
mrms = MRMSData( )
obj_id = ObjectIdentification( )
get_time = personal_datetime( )
obj_match = ObjectMatching( dist_max = 6 ) 
name = 'Rotation Objects (QC9)'
######### LSRS #########
from skimage.measure import regionprops
from loadLSRs import loadLSR
from loadWWAs import loadWWA

kwargs = {'cblabel': 'Reflectivity', 'alpha':0.7, 'extend': 'neither', 'cmap': 'dbz' } 
plt = Plotting( date=date, z1_levels = np.arange(5,80, 5), z2_levels = [0.], z3_levels = [0.], **kwargs )
fig, ax, map_ax, x, y = plt._create_fig( fig_num = 0, plot_map = True, figsize = (8, 9) )

valid_date_and_time, initial_date_and_time = get_time.determine_forecast_valid_datetime( date_dir = date, time_dir = time, fcst_time_idx = fcst_time_idx )
storm_objects_40, storm_objects_35, dbz = mrms.load_mrms_dbz( date_dir = date, valid_datetime = valid_date_and_time, tag='single_thresh_only' )
rot_tracks, az_shear = mrms.load_az_shr_tracks( var = var_mrms, name=name, date_dir = date, valid_datetime = valid_date_and_time )

load_lsr = loadLSR(date_dir=date, date=initial_date_and_time[0], time=initial_date_and_time[1])
load_wwa = loadWWA(date_dir=date, date=initial_date_and_time[0], time=initial_date_and_time[1], time_window = 15)
hail_ll = load_lsr.load_hail_reports( )
torn_ll = load_lsr.load_tornado_reports( )
wind_ll = load_lsr.load_wind_reports( )
torn_wwa_ll = load_wwa.load_tornado_warning_polygon( return_polygons=True )

lsr_points = {'hail': hail_ll, 'tornado': torn_ll, 'wind': wind_ll } 
wwa_points = {'tornado': torn_wwa_ll }

'''
lsr_lons = np.concatenate((hail_ll[1], torn_ll[1], wind_ll[1]))
lsr_lats = np.concatenate((hail_ll[0], torn_ll[0], wind_ll[0]))
lsr_xy = load_lsr.to_xy( lats=lsr_lats, lons=lsr_lons)
lsr_xy = list(zip(lsr_xy[1,:],lsr_xy[0,:]))
qc_params = {'lsr':{'lsr_dist': 10, 'lsr_points': lsr_xy}}
qc_objects, _ = obj_id.quality_control(object_labels=rot_tracks, object_props=regionprops(rot_tracks,az_shear), input_data=az_shear, **qc_params)
'''

plt.spatial_plotting( fig=fig, ax=ax, map_ax = map_ax[0], x=x, y=y, z1=dbz, z2=rot_tracks, lsr_points=lsr_points, wwa_points=wwa_points, title='')

fname = 'rot_track_with_dbz_%s_%s.png' % ( initial_date_and_time[0], initial_date_and_time[1]) 
plt._save_fig( fig=fig, fname = fname)

