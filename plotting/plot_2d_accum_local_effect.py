from os.path import join
import matplotlib.pyplot as plt
from wofs.util.news_e_plotting_cbook_v2 import cb_colors as wofs
import sys
sys.path.append('/home/monte.flora/machine_learning/build_model')
from ModelClarifier.class_ModelClarify import ModelClarify
from ModelClarifier.plotting_routines import plot_first_order_ale, plot_second_order_ale
from wofs.util import feature_names

from MachLearn import MachLearn
ml = MachLearn()

fname_params = {
           'model_name' : 'RandomForest',
           'target_var': 'matched_to_tornado_0km',
           'resampling_method': 'random',
           'fcst_time_idx': 'first_hour' 
           }

variables_to_remove = [
                    'matched_to_severe_wx_warn_polys_15km', 'matched_to_severe_wx_warn_polys_30km',
                    'matched_to_tornado_warn_ploys_15km', 'matched_to_tornado_warn_ploys_30km']

print ('Loading the data...')
data_dict = ml.load_cv_data(
                            fname_params=fname_params, 
                            variables_to_remove=variables_to_remove, 
                            load_model=True,
                            debug=True,
                            fold_to_load = 2
                            )

line_colors = [wofs.red8, wofs.blue8, wofs.gray8, wofs.green8, wofs.orange8, wofs.purple8, 'k', wofs.blue5]
kwargs = {'line_color': wofs.red8, 'facecolor': wofs.blue5}
fold = 'fold_2'

model = data_dict[fold]['model']
data = data_dict[fold]['data']
examples = data['validation']['examples']
targets = data['validation']['targets']

model_clarifier = ModelClarify( 
                                model=model,
                                examples_in = examples,
                                targets_in = targets,
                                )

result = model_clarifier.get_top_contributors()

hit_result = result['hits']
variable_list = list(hit_result.keys())

top_feature = variable_list[0]
for f in variable_list[1:]:
    features = [top_feature, f]
    print(features)
    #print('Calculating the ALE for {}...'.format(variable))
    ale, quantiles = model_clarifier.calculate_second_order_ale(features=features)
    fig, ax = plt.subplots(figsize=(6,6))
    plot_second_order_ale(ax=ax, ale_data=ale, quantile_tuple=quantiles, feature_names=features, **kwargs)
    plt.savefig(f'ale_second_order_{features[0]}_{features[1]}_{fname_params["model_name"]}_{fname_params["target_var"]}.png', bbox_inches='tight')
        
