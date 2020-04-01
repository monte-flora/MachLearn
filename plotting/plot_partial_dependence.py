import numpy as np 
from os.path import join
import matplotlib.pyplot as plt
from wofs.util.news_e_plotting_cbook_v2 import cb_colors as wofs
import sys
sys.path.append('/home/monte.flora/machine_learning/build_model')
from ModelClarifier.class_ModelClarify import ModelClarify
from ModelClarifier.plotting_routines import plot_pdp_1d
from wofs.util import feature_names

from MachLearn import MachLearn

fold_to_load = 1
fname_params = {
    "model_name": "RandomForest",
    "target_var": "matched_to_tornado_0km",
    "fcst_time_idx": "first_hour",
}

print("Loading the data...")
ml = MachLearn(
    fname_params,
    load_model=True,
    drop_correlated_features=True,
    fold_to_load=fold_to_load,
)
data_dict = ml.data_dict

fold = f"fold_{fold_to_load}"
model = data_dict[fold]["model"]
data = data_dict[fold]["data"]
examples = data["validation"]["examples"]
targets = data["validation"]["targets"]

print(len(data["validation"]["feature_names"]))
print(len(data["training"]["feature_names"]))

model_clarifier = ModelClarify(model=model, examples_in=examples, targets_in=targets,)

line_colors = [wofs.red8, wofs.blue8, wofs.gray8, wofs.green8, wofs.orange8, wofs.purple8, 'k', wofs.blue5]
kwargs = {'line_color': wofs.red8, 'facecolor': wofs.blue5}

result = model_clarifier.get_top_contributors()

hit_result = result['hits']
variable_list = list(hit_result.keys())

for variable in variable_list:
    print('Calculating the ALE for {}...'.format(variable))
    pdp, quantiles = model_clarifier.compute_1d_partial_dependence(feature=variable)
    fig, ax = plt.subplots(figsize=(8,4))
    plot_pdp_1d(ax=ax, pdp_data=pdp, quantiles=quantiles, feature_name=variable, examples=examples, **kwargs)
    plt.savefig(f'pdp_{variable}_{fname_params["model_name"]}_{fname_params["target_var"]}.png', bbox_inches='tight')
