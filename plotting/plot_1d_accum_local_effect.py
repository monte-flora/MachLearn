import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from wofs.util.news_e_plotting_cbook_v2 import cb_colors as wofs
import sys

sys.path.append("/home/monte.flora/machine_learning/build_model")
from ModelClarifier.class_ModelClarify import ModelClarify
from ModelClarifier.plotting_routines import plot_first_order_ale
from wofs.util import feature_names
from MachLearn import MachLearn
import pickle
from joblib import load

path = '/work/mflora/ML_DATA/MODEL_SAVES/OPERATIONAL'

ml = MachLearn(drop_correlated_features=True, preload=False)
for model_name in ['RandomForest', 'LogisticRegression']:
    for target in ['matched_to_tornado_0km', 'matched_to_severe_hail_0km', 'matched_to_severe_wind_0km']:
        data = ml.load_dataframe(
            target_var_name=target,
            load_filename_dict={
                "training": f"/work/mflora/ML_DATA/DATA/operational_training_first_hour_resampled_to_{target}_dataset.pkl"
            },
            additional_vars_to_drop=ml.additional_vars_to_drop,
        )

        if model_name == 'LogisticRegression':
            data = ml.normalize(data, '')

        ml_model = load(join(path, f'{model_name}_{target}.pkl'))
        examples = data['training']['examples']
        targets = data['training']['targets']

        mc = ModelClarify(model=ml_model, examples_in=examples, targets_in=targets)

        results_fname = f'permutation_importance_{model_name}_{target}.pkl'

        print(f'Loading {results_fname}...')
        with open(join(path,results_fname), 'rb') as pkl_file:
            results = pickle.load(pkl_file)

        variable_list = list(results.retrieve_multipass().keys())

        for variable in variable_list:
            print("Calculating the ALE for {}...".format(variable))
            ale, quantiles = mc.calc_ale(feature=variable, nbootstrap=1, subsample=1.0)
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_first_order_ale(
                ax=ax,
                ale_data=np.array(ale),
                quantiles=quantiles,
                feature_name=variable,
                )
            plt.savefig(
                    f'ale_{variable}_{model_name}_{target}.png',
                    bbox_inches="tight",
                    )
