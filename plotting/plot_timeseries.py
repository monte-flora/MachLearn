from wofs.plotting.Plot import Plotting
from wofs.plotting.Plot import verification_plots, plot_timeseries
from load_results import load
import wofs.plotting.plotting_config as plt_config
import numpy as np
import argparse
from collections import ChainMap

plt = Plotting()

""" 
 Usage: 
     python plot_timeseries.py --target matched_to_tornado_0km 
"""

title_dict = {
    "matched_to_tornado_0km": "Tornadoes",
    "matched_to_severe_hail_0km": "Severe Hail",
    "matched_to_severe_wind_0km": "Severe Wind",
}

# Setting up the parser
parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str)
args = parser.parse_args()

verify_var = args.target

ml_model_names = ["RandomForest", 'XGBoost', 'LogisticRegression']

fcst_time_idx_set = [0, 6, 12, 18,24]
adict = {
    "mode": ("different_times", fcst_time_idx_set),
    "verify_var": verify_var,
    "vars_to_load": ["pod", "sr", "auprc", "auc", "bss"],
    "correlated_feature_opt": True,
}
fig_fname = f"timeseries_of_multiple_metrics_{verify_var}.png"
#################

results = []
for model in ml_model_names:
    adict['model_name'] = model
    results.append(load(adict))

results = dict(ChainMap(*results))
print(results.keys())

def calc_csi(pod, sr):
    """
    Critical Success Index.
    Formula: Hits / ( Hits+Misses+FalseAlarms)
    """
    sr[np.where(sr == 0)] = 1e-5
    pod[np.where(pod == 0)] = 1e-5
    return 1.0 / ((1.0 / sr) + (1.0 / pod) - 1.0)


metrics = ["max_csi", "auc", "auprc", "bss"]
metrics_nice = {"max_csi": "Maximum CSI", "auc": "AUC", "auprc": "AUPRC", "bss": "BSS"}

new_results = {metrics_nice[metric]: {} for metric in metrics}
for metric in metrics:
    for model in ml_model_names:
        if metric == "max_csi":
            new_results[metrics_nice[metric]][model] = [
                np.mean(
                    np.amax(
                        calc_csi(
                            results[model][i]["pod"],
                            results[model][i]["sr"],
                        ),
                        axis=1,
                    )
                )
                for i, _ in enumerate(fcst_time_idx_set)
            ]
        else:
            new_results[metrics_nice[metric]][model] = [
                np.mean(results[model][i][metric])
                for i, _ in enumerate(fcst_time_idx_set)
            ]

title = f"{title_dict[verify_var]} \n performance for specific forecast periods"
line_colors = plt_config.colors_for_ml_models_dict
fig = plot_timeseries(
    fcst_time_idx_set,
    new_results,
    line_labels=ml_model_names,
    line_colors=line_colors,
    title=title,
)

plt._save_fig(fig=fig, fname=fig_fname)
