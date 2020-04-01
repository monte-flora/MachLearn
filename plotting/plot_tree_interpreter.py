from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from wofs.util.news_e_plotting_cbook_v2 import cb_colors as wofs
import sys

sys.path.append("/home/monte.flora/machine_learning/build_model")
from ModelClarifier.class_ModelClarify import ModelClarify
from ModelClarifier.plotting_routines import plot_first_order_ale
import waterfall_chart
from MachLearn import MachLearn
from wofs.util.feature_names import to_only_varname

"""
fold_to_load=None
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

results_list = [ ]
for fold in range(15):
    fold = f"fold_{fold}"
    model = data_dict[fold]["model"]
    data = data_dict[fold]["data"]
    examples = data['training']["examples"]
    targets = data['training']["targets"]
    model_clarifier = ModelClarify(model=model, examples_in=examples, targets_in=targets)
    result = model_clarifier.get_top_contributors()
    results_list.append(result)

for mode in ['hits', 'misses', 'false_alarms', 'corr_negs']:
    for var in list(results_list[0][mode].keys()):
        result[mode][var] = {'Mean Contribution': np.mean([adict[mode][var]['Mean Contribution'] for adict in results_list])}

print(result)
"""

def combine_like_features(contrib, varnames):
    """
    """
    duplicate_vars = {}
    for var in varnames:
        duplicate_vars[var] = [idx for idx, v in enumerate(varnames) if v == var]

    new_contrib = []
    new_varnames = []
    for var in list(duplicate_vars.keys()):
        idxs = duplicate_vars[var]
        new_varnames.append(var)
        new_contrib.append(np.array(contrib)[idxs].sum())

    return new_contrib, new_varnames


def plot_treeinterpret(result, save_name):
    """
    Plot the results of tree interpret
    """
    contrib = [50.0]
    varnames = ["bias"]
    for i, var in enumerate(list(result.keys())):
        contrib.append(result[var]["Mean Contribution"])
        varnames.append(to_only_varname(var))

    contrib, varnames = combine_like_features(contrib, varnames)

    plt = waterfall_chart.plot(
        varnames,
        contrib,
        rotation_value=90,
        sorted_value=True,
        threshold=0.02,
        net_label="Final prediction",
        other_label="Others",
        y_lab="Probability",
    )
    plt.savefig(save_name, bbox_inches="tight", dpi=300)


#for mode in ["hits", "misses", "false_alarms", "corr_negs"]:
#    plot_treeinterpret(result[mode], save_name="tree_interpreter_{}.png".format(mode))
