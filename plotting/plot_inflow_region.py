import sys
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/wofs/plotting')
sys.path.append('/home/monte.flora/wofs/data')
sys.path.append('/home/monte.flora/machine_learning/object_based_learning/extraction')
import config
from Plot import Plotting
import xarray as xr
from loadEnsembleData import EnsembleData
from loadEnsembleData import calc_time_max
from StormBasedFeatureEngineering import StormBasedFeatureEngineering
from os.path import join
from basic_functions import personal_datetime
import numpy as np 
from StoringObjectProperties import initialize_dict
import itertools

################################################################################
''' Plot storm inflow regions on the hour-long mesocyclone tracks '''
#################################################################################

date = '20170516'
time = '0000'
fcst_time_idx = 6
var_mrms = 'LOW_CRESSMAN'
var_newse = 'uh_0to2'
extract = StormBasedFeatureEngineering( )
property_name_list = initialize_dict( )

def _extract_storm_features( input_data, x_object_cent, y_object_cent, ROI_STORM=7 ):
        ''' Extract intra-storm state features for machine learning '''
        mask = np.zeros(( input_data.shape[-2], input_data.shape[-1] ) )
        x = np.arange(input_data.shape[-1])
        y = np.arange(input_data.shape[-2])
        stat_functions = [ (np.mean, None) ]
        object_centroids = list(zip( y_object_cent, x_object_cent ))
        obj_strm_data = np.zeros(( len(object_centroids), input_data.shape[0] * len(stat_functions) ))
        for i, obj_cent in enumerate( object_centroids ):
            rho, phi = _cart2pol( x[np.newaxis,:]-obj_cent[1], y[:,np.newaxis]-obj_cent[0] )
            storm_points = np.where(rho <= ROI_STORM )
            for j, k in enumerate( list(itertools.product( range(input_data.shape[0]), range(len(stat_functions)) ))):
                v = k[0] ; s = k[1]
                temp_data = input_data[v,:,:]
                func_set = stat_functions[s]
                obj_strm_data[i, j] = np.mean( temp_data[storm_points] )
            mask[storm_points] = 10.    

        return obj_strm_data, mask #dim = (n_objects, n_var*n_stats)

def _find_storm_inflow_region( bunk_v, bunk_u, rho, phi, ROI_ENV = 15  ):
        ''' Find storm inflow region using the average intra-storm bunker's motion vector '''
        # Bunker's motion in degrees
        left = ( np.arctan2( bunk_v, bunk_u ) * (180./np.pi) ) + 10.
        right = left - 110.
        inflow_indices = np.where((phi <= left )& (phi >= right)&(rho <= ROI_ENV ))

        return inflow_indices

def _cart2pol( x, y ):
        ''' Converts from cartesian coordinates to polar coordinates '''
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x) * (180./np.pi)
        return(rho, phi)

def _mask_inflow_region( input_data, x_object_cent, y_object_cent, avg_bunk_v_per_obj, avg_bunk_u_per_obj ):
        ''' Extract storm-inflow environment features for machine learning '''
        x = np.arange(input_data.shape[-1])
        y = np.arange(input_data.shape[-2])
        mask = np.zeros(( input_data.shape[-2], input_data.shape[-1] ) )
        object_centroids = list(zip( y_object_cent, x_object_cent ))
        for i, obj_cent in enumerate( object_centroids ):
            rho, phi = _cart2pol( x[np.newaxis,:]-obj_cent[1], y[:,np.newaxis]-obj_cent[0] )
            bunk_u = avg_bunk_u_per_obj[i]
            bunk_v = avg_bunk_v_per_obj[i]
            env_points = _find_storm_inflow_region( bunk_u, bunk_v, rho, phi )
            mask[env_points] = 1.0

        return mask 

# Load in the bunker's motion vectors 
time_indexs_env = [ 12 ] # valid at 60 min
storm_motion_vars = [ 'bunk_r_u', 'bunk_r_v' ]
data_smryfiles = EnsembleData( date_dir =date, time_dir = time, base_path ='summary_files')
storm_motion_data = data_smryfiles.load( var_name_list=storm_motion_vars, time_indexs=time_indexs_env, tag='ENV' )
storm_motion_data_time_max = calc_time_max( storm_motion_data )

print np.shape( storm_motion_data_time_max ) 

# Load object properties dataframe for the forecast storm objects 
in_path = join( config.object_storage_dir_ml, date )
title = 'MESO'
object_file = join(in_path, 'HOUR-LONG_WOFS%s_OBJECTS_%s-%s_%02d.nc' % (title, date, time, fcst_time_idx))
ds = xr.open_dataset( object_file )
ds_i = ds[property_name_list]
df = ds_i.to_dataframe()
mem_idx = 15 
df_at_mem_idx  = df.loc[ df['Ensemble member'] == mem_idx]
x_object_cent = np.rint( df_at_mem_idx['obj_centroid_x'].values ).astype(int)
y_object_cent = np.rint( df_at_mem_idx['obj_centroid_y'].values ).astype(int)

good_idx = extract._remove_objects_near_boundary( x_object_cent, y_object_cent, NY=storm_motion_data.shape[-2], NX=storm_motion_data.shape[-1] )
good_x_cent = x_object_cent[good_idx]
good_y_cent = y_object_cent[good_idx]

data_strm, strm_mask = _extract_storm_features( storm_motion_data_time_max[:, mem_idx, :,:], good_x_cent, good_y_cent )
print np.shape( data_strm )

mask = _mask_inflow_region( storm_motion_data_time_max[:, mem_idx, :,:], good_x_cent, good_y_cent, avg_bunk_v_per_obj=data_strm[:,1], avg_bunk_u_per_obj=data_strm[:,0] )
rot_tracks = ds['Objects'].values
rot_tracks = rot_tracks[mem_idx,:,:]

kwargs = {'cblabel': 'Reflectivity', 'alpha':0.7, 'extend': 'neither', 'cmap': 'basic' }
plt = Plotting( date=date, z1_levels = [0.1, 2], z2_levels = [0.], z3_levels = [0.], **kwargs ) 
fig, ax, map_ax, x, y = plt._create_fig( fig_num = 0, plot_map = True, figsize = (8, 9) )

plt.spatial_plotting( fig=fig, ax=ax, map_ax = map_ax[0], x=x, y=y, z1=mask, z2=rot_tracks, z3=strm_mask, title='' )

fname = 'inflow_region_%s_%s.png' % ( date, time ) 
plt._save_fig( fig=fig, fname = fname)

