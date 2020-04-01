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
     python plot_perform_metrics.py --target matched_to_tornado_0km 
"""

ml_model_names = ["RandomForest", 'XGBoost', 'LogisticRegression']

adict = {
    "mode": ("different_models", ml_model_names),
    "vars_to_load": ["pod", "sr"],
    "fcst_time_idx": 'first_hour',
    "correlated_feature_opt": True
}
#################

def calc_csi(pod, sr):
    """
    Critical Success Index.
    Formula: Hits / ( Hits+Misses+FalseAlarms)
    """
    sr[np.where(sr == 0)] = 1e-5
    pod[np.where(pod == 0)] = 1e-5
    return 1.0 / ((1.0 / sr) + (1.0 / pod) - 1.0)

for target in ['matched_to_tornado_0km', 'matched_to_severe_hail_0km', 'matched_to_severe_wind_0km']:
    adict['verify_var'] = target
    results = load(adict)

    for model in ml_model_names:
        pod = np.mean(results[model]['pod'], axis=0)
        sr = np.mean(results[model]['sr'], axis=0)
        csi = calc_csi(pod, sr)
        bias = pod/sr
        idx = np.argmax(csi)
        print( f'\n {model} {target}......Max CSI: {csi[idx]} and bias: {bias[idx]}')



