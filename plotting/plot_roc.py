from wofs.util import config
from wofs.plotting.Plot import Plotting 
from wofs.plotting.Plot import verification_plots
import wofs.plotting.plotting_config as plt_config 
import argparse
from load_results import load
plt = Plotting(  )
verification_plots = verification_plots()

""" usage: python plot_roc.py -m ['RandomForest', 'XGBoost'] -t 0 --target matched_to_tornado_0km """

title_dict = {'matched_to_tornado_0km':'Tornadoes',
              'matched_to_severe_hail_0km':'Severe Hail',
              'matched_to_severe_wind_0km':'Severe Wind'
              }

# Setting up the parser 
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models", nargs='+', help='<Required> Set flag', required=True)
parser.add_argument("-t", "--fcst_time_idx", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("-c", '--correlated', type=int)
args = parser.parse_args()

ml_model_names = list([ m.replace('[','').replace(']','').replace(',','') for m in args.models])
fcst_time_idx = args.fcst_time_idx
verify_var = args.target
correlated = args.correlated
if correlated == 1:
    correlated_feature_opt = True
else:
    correlated_feature_opt = False

corr_title_dict = {True: 'Correlated Features Removed',
                   False: 'All Features'
                  }

# Don't need to change!!!
fig, axes = plt._create_fig( fig_num = 0, figsize = (8, 8))
if len(ml_model_names) >= 2:
    adict = {
         'mode': ('different_models', ml_model_names),
         'fcst_time_idx': fcst_time_idx,
         'verify_var': verify_var,
         'vars_to_load': ['pod', 'pofd', 'auc'],
         'correlated_feature_opt': correlated_feature_opt
         }
    line_colors = plt_config.colors_for_ml_models_dict
    line_labels = ml_model_names
    fig_fname = 'roc_diagram_all_models_{}_{}_rmv_corr={}.png'.format(verify_var, fcst_time_idx, correlated_feature_opt)

else:
    fcst_time_idx_set = [0,6,12]
    adict = {
         'mode': ('different_times', fcst_time_idx_set),
         'model_name': ml_model_names[0],
         'verify_var': verify_var,
         'vars_to_load': ['pod', 'pofd', 'auc'],
         'correlated_feature_opt': correlated_feature_opt
         }
    line_labels = plt_config.get_line_labels( ml_model_names[0], fcst_time_idx_set)
    line_colors = plt_config.colors_per_time( line_labels )
    fig_fname = 'roc_diagram_{}_{}_{}_rmv_corr={}.png'.format(ml_model_names[0], verify_var, fcst_time_idx, correlated_feature_opt)

results = load( adict )

verification_plots.plot_roc_curve( ax = axes,
            results=results, 
            line_colors = line_colors, 
            line_labels = line_labels,
            subpanel_labels = ('', ''), 
            counter = 2,
            title=f'{title_dict[verify_var]} \n 30-min lead time; averaged over forecasts initialized within the {fcst_time_idx.replace("_", " ")} \n {corr_title_dict[correlated_feature_opt]}'
            )
plt._add_major_frame( fig, 
                      fontsize=25, 
                      xlabel_str='Probability of False Detection (POFD)', 
                      ylabel_str='Probability of Detection (POD)', title = '' )
plt._save_fig( fig=fig, fname = fig_fname)
