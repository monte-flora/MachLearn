import sys
sys.path.append("/home/monte.flora/machine_learning/build_model")
from MachLearn import MachLearn
import pandas as pd
from os.path import join
import pickle
from joblib import dump, load
from ModelClarifier.class_ModelClarify import ModelClarify

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

        results = mc.permutation_importance(n_multipass_vars=15, evaluation_fn='auc', subsample=1.0, njobs=0.4, nbootstrap=1000)
        
        results_fname = f'permutation_importance_{model_name}_{target}.pkl'
        
        print(f'Saving {results_fname}...')
        with open(join(path,results_fname), 'wb') as pkl_file:
            pickle.dump(results, pkl_file)



