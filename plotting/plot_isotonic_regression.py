from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

model_names = ['RandomForest', 'XGBoost', 'LogisticRegression']
target_vars = ['matched_to_tornado_0km', 'matched_to_severe_hail_0km', 'matched_to_severe_wind_0km']
fcst_time_idx = 'first_hour'


def plot_isotonic_regression_curve( model, target_var, fcst_time_idx='first_hour'):
    """
    Plot the learned relationship between calibrated and uncalibrated prediction 
    """
    predictions = np.arange(0, 1.05, 0.05)
    plt.figure(figsize=(6,6))
    plt.plot(predictions, predictions, linestyle='dashed', alpha=0.5, color='k')
    
    rcParams["xtick.labelsize"] = 15
    rcParams["ytick.labelsize"] = 15

    all_curves = [ ]
    for f in range(15):
        isofile = f'/work/mflora/ML_DATA/MODEL_SAVES/FCST_TIME_IDX={fcst_time_idx}/{model}/model:{model}_isotonic_correlated_features_removed:{target_var}_fold:{f}.joblib'
        temp_file = isofile.format(f)
        clf = load(temp_file)
        calibrated_predictions = clf.transform(predictions)
        all_curves.append(calibrated_predictions)

    mean_curve = np.mean(all_curves, axis=0)
    top_curve = np.percentile(all_curves, 97.5, axis=0)
    bottom_curve = np.percentile(all_curves, 2.5, axis=0)
    
    plt.plot(predictions, mean_curve, linewidth=2.2, alpha=0.7, color='r')
    plt.fill_between(predictions, top_curve, bottom_curve, alpha=0.4, color='r')

    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.ylabel('Calibrated Prediction', fontsize=12, alpha=0.7)
    plt.xlabel('Original Prediction', fontsize=12, alpha=0.7)
    plt.grid(alpha=0.5)
    plt.title(f'{model} {target_var}', fontsize=12, alpha=0.7)
    plt.savefig(f'isotonic_regression_{model}_{target_var}.png', bbox_inches='tight')

for model in model_names:
    print(model)
    for target_var in target_vars:
        plot_isotonic_regression_curve( model, target_var, fcst_time_idx='first_hour')

