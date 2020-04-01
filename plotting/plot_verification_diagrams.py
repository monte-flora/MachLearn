from wofs.plotting.Plot import Plotting 
from wofs.plotting.Plot import verification_plots
from load_results import load
import wofs.plotting.plotting_config as plt_config
import numpy as np 
import argparse
plt = Plotting(  )

""" usage: python plot_verification_diagrams.py -m LogisticRegression -t first_hour --target matched_to_tornado_0km -c True """

# Setting up the parser 
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models", nargs='+', help='<Required> Set flag', required=True)
parser.add_argument("-t", "--fcst_time_idx", type=str)
parser.add_argument('-c', '--correlated_feature_opt', type=int)
parser.add_argument("--target", type=str)
args = parser.parse_args()

ml_model_names = list([ m.replace('[','').replace(']','').replace(',','') for m in args.models])
fcst_time_idx = args.fcst_time_idx
verify_var = args.target
correlated_feature_opt = args.correlated_feature_opt
plot_opt = 'different_features'

if correlated_feature_opt == 1:
    correlated_feature_opt = True
else:
    correlated_feature_opt = False

if plot_opt == 'different_models':
    adict = { 
         'mode': ('different_models', ml_model_names),
         'fcst_time_idx': fcst_time_idx,
         'verify_var': verify_var,
         'vars_to_load': ['pod', 'sr', 'auprc', 'targets', 'mean fcst prob', 'event frequency', 'bss', 'predictions', 'pofd', 'auc'],
         'correlated_feature_opt': correlated_feature_opt
         }
    line_colors = plt_config.colors_for_ml_models_dict
    line_labels = ml_model_names
    fig_fname = 'verification_diagram_all_models_{}_{}_{}.png'.format(verify_var, fcst_time_idx, correlated_feature_opt)

elif plot_opt == 'different_times':
    fcst_time_idx_set = [0,6,12]
    adict = { 
         'mode': ('different_times', fcst_time_idx_set),
         'model_name': ml_model_names[0],
         'verify_var': verify_var,
         'vars_to_load': ['pod', 'sr', 'auprc', 'targets', 'mean fcst prob', 'event frequency', 'bss', 'predictions', 'pofd', 'auc'],
         'correlated_feature_opt': correlated_feature_opt
         }

    line_labels =  plt_config.colors_for_ml_models_dict
    line_colors = plt_config.colors_per_time( line_labels )
    fig_fname = 'verification_diagram_all_times_{}_{}_{}.png'.format(ml_model_names[0], verify_var, correlated_feature_opt)

elif plot_opt == 'different_features':
    adict = { 
         'mode': ('different_features', [True, False]),
         'model_name': ml_model_names[0],
         'verify_var': verify_var,
         'fcst_time_idx': fcst_time_idx,
         'vars_to_load': ['pod', 'sr', 'auprc', 'targets', 'mean fcst prob', 'event frequency', 'bss', 'predictions', 'pofd', 'auc'],
         }
    line_labels = {f'{ml_model_names[0]} correlated_features_removed': f"{ml_model_names[0]} removed correlated features", f'{ml_model_names[0]} all_features': f"{ml_model_names[0]} all features"}
    line_colors = {f'{ml_model_names[0]} correlated_features_removed': 'r', f'{ml_model_names[0]} all_features': 'k'}
    fig_fname = 'verification_diagram_diff_features_{}_{}_{}.png'.format(ml_model_names[0], verify_var, fcst_time_idx)


#################
results = load( adict )
fig, axes = plt._create_fig( fig_num = 0, figsize = (16, 8), sub_plots=(1,3))
# Plot ROC Diagram
verification_plots.plot_roc_curve( ax = axes.flat[0],
            results=results,
            line_colors = line_colors,
            line_labels = line_labels,
            subpanel_labels = ('', ''),
            counter = 2)

axes.flat[0].set_ylabel('Probability of Detection (POD)')
axes.flat[0].set_xlabel('Probability of False Detection (POFD)')

# Plot attribute diagram
verification_plots.plot_attribute_diagram( ax = axes.flat[1],
                results=results,
                line_colors = line_colors,
                line_labels = line_labels,
                subpanel_labels = ('',''),
                counter= 2,
                inset_loc='upper left',
                bin_rng = np.round( np.arange(0, 1., 0.1), 5 ))

axes.flat[1].set_ylabel('Observed Frequency')
axes.flat[1].set_xlabel('Mean Forecast Probability')

# Plot Performance diagram
csiLines = verification_plots.plot_performance_diagram( ax = axes.flat[2],
            results = results, 
            line_colors = line_colors, 
            line_labels = line_labels,
            subpanel_labels = ('', ''), 
            counter = 2)

axes.flat[2].set_ylabel('Probability of Detection (POD)')
axes.flat[2].set_xlabel('Success Ratio (1-FAR)')

plt._add_major_colobar( fig, 
                        contours=csiLines, 
                        label = 'Critical Success Index', 
                        tick_fontsize = 15,
                        fontsize = 25, 
                        labelpad=  35, 
                        coords = [0.92, 0.11, 0.03, 0.77] )

plt._save_fig( fig=fig, fname = fig_fname)
