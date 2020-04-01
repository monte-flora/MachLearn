import os 
from os.path import join 
from wofs.util import config
from wofs.plotting.Plot import Plotting
import xarray as xr
from wofs.data.loadMRMSData import MRMSData
from wofs.util.basic_functions import personal_datetime
import numpy as np 
from wofs.data.loadLSRs import loadLSR
from wofs.data.loadWWAs import loadWWA, is_severe, load_reports
get_time = personal_datetime( )
from wofs.processing.ObjectMatching import ObjectMatching, match_to_lsrs
from skimage.measure import regionprops
import pandas as pd
from machine_learning.main.predict import worker

from ModelClarifier.class_ModelClarify import ModelClarify
from plot_tree_interpreter import plot_treeinterpret

date = '20190507'
time = '0030'
fcst_time_idx = 6
target = 'matched_to_severe_hail_0km'

ncfname = f'/work/mflora/ML_DATA/MODEL_OBJECTS/{date}/updraft_ensemble_objects_{date}-{time}_t:{fcst_time_idx}.nc'

title_dict = {'matched_to_tornado_0km':'Tornadoes',
              'matched_to_severe_hail_0km':'Severe Hail',
              'matched_to_severe_wind_0km':'Severe Wind'
              }

ds = xr.open_dataset(ncfname)
probability_objects = ds['Probability Objects'].values
prob_of_storm_location = ds['2D Probabilities'].values

valid_date_and_time, initial_date_and_time = get_time.determine_forecast_valid_datetime( date_dir=str(date), time_dir=time, fcst_time_idx=fcst_time_idx )

load_lsr = loadLSR(date_dir=date, date=valid_date_and_time[0], time=valid_date_and_time[1], forecast_length = 30)
hail_ll = load_lsr.load_hail_reports( )
torn_ll = load_lsr.load_tornado_reports( )
wind_ll = load_lsr.load_wind_reports( )
hail_xy, torn_xy, wind_xy = load_reports(str(date), valid_date_and_time, time_window=15, forecast_length = 30)
lsr_points = {'hail': hail_ll, 'tornado': torn_ll, 'wind': wind_ll }

object_props = regionprops(probability_objects.astype(int), prob_of_storm_location, coordinates='rc' )
verification_dict = {
                     'severe_hail': hail_xy,
                     'severe_wind': wind_xy,
                     'tornado': torn_xy
                     }
matched_at_15km = { 'matched_to_{}_15km'.format(atype):
                     match_to_lsrs( object_properties=object_props,
                     lsr_points=verification_dict[atype], dist_to_lsr=1 ) for atype in verification_dict.keys() }

matched_at_15km = is_severe(matched_at_15km, '15km')
matched_to_lsrs = matched_at_15km['matched_to_LSRs_15km']

probability_objects_labelled_severe = np.zeros(( probability_objects.shape ))
for region in object_props:
    if matched_to_lsrs[region.label] == 1:
        probability_objects_labelled_severe[probability_objects==region.label] = 1 
    else:
        probability_objects_labelled_severe[probability_objects==region.label] = -1 


kwargs = {'alpha':0.7, 'extend': 'neither', 'cmap': 'wofs' }
plt = Plotting( date=date, z1_levels = np.arange(0.05, 1.15, .1), z2_levels = [0.], z3_levels=[0.], **kwargs )
fig, axes, map_axes, x, y = plt._create_fig( 
                                            fig_num = 0, 
                                            sub_plots =(2,3), 
                                            plot_map = True, 
                                            figsize = (12, 8), 
                                            )

def title_inside_fig(ax, title):
    ax.text(.5,.9,f'{title}',
        horizontalalignment='center',
        transform=ax.transAxes, 
        fontsize=10)

# Upper left hand : Probability of Storm Location
plt.spatial_plotting(fig, axes.flat[0], x, y, z1=prob_of_storm_location, map_ax=map_axes[0])
#axes.flat[0].set_title('Probability of Storm Location')
title_inside_fig(axes.flat[0], 'Probability of Storm Location')

# Upper right hand: Identified Probability Objects  
kwargs = {'alpha':0.7, 'extend': 'neither', 'cmap': 'qualitative' }
plt = Plotting( date=date, z1_levels = np.arange(1,np.amax(probability_objects)+1), z2_levels = [0.], z3_levels=[0.], **kwargs )
plt.spatial_plotting(fig, axes.flat[1], x, y, z1=np.ma.masked_where(probability_objects<0.001, probability_objects), map_ax=map_axes[1], z1_is_integers=True)
title_inside_fig(axes.flat[1], 'Probability Objects')

# Lower left hand: Matched (Unmatched) to an LSR
kwargs = {'alpha':0.7, 'extend': 'neither', 'cmap': 'diverge' }
plt = Plotting( date=date, z1_levels = [-1, 0, 1], z2_levels = [0.], z3_levels=[0.], **kwargs ) 
plt.spatial_plotting(fig, axes.flat[2], x, y, z1=np.ma.masked_where(probability_objects_labelled_severe==0., probability_objects_labelled_severe), lsr_points=lsr_points, map_ax=map_axes[2], z1_is_integers=True)
title_inside_fig(axes.flat[2], 'Matched to an LSR')

#--------------------------------------------------------------------
# Lower right hand: Forecast Probability from A ML Model
kwargs = {'alpha':0.7, 'extend': 'neither', 'cmap': 'wofs' }
plt = Plotting( date=date, z1_levels = np.arange(0., 0.65, 0.025), z2_levels = [0.], z3_levels=[0.], **kwargs )

for i, model in zip([3,4,5], ['RandomForest', 'XGBoost', 'LogisticRegression']):
    data = worker(date, time, fcst_time_idx, model, target, return_data=True)
    forecast_probabilities = data['probs_2d']
    contours = plt.spatial_plotting(fig, axes.flat[i], x, y, z1=np.ma.masked_where(forecast_probabilities==0., forecast_probabilities), lsr_points=lsr_points, map_ax=map_axes[i])
    title_inside_fig(axes.flat[i], f'{model}')

#plt._add_major_frame(fig=fig,
#                     fontsize= 12,
#                     title=f'{title_dict[target]}\n Init Time: {initial_date_and_time[0]}-{initial_date_and_time[1]} \n Valid Time: {valid_date_and_time[0]}-{valid_date_and_time[1]} ')
#plt._add_major_colobar(fig, contours, label='Probability', fontsize=15, coords=[0.92, 0.31, 0.03, 0.22], labelpad=25)
#
#fig.subplots_adjust(hspace=.002)
#
fname = f'predictions_all_models_{target}_{date}-{time}_t:{fcst_time_idx}.png'
plt._save_fig( fig=fig, fname = fname)

