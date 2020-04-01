import sys
sys.path.append("/home/monte.flora/machine_learning/build_model")
from MachLearn import MachLearn
import pandas as pd
from os.path import join
import pickle
from joblib import dump, load

model_name = 'RandomForest'
target = 'matched_to_tornado_0km'
path = '/work/mflora/ML_DATA/MODEL_SAVES/OPERATIONAL'

def return_params(model_name, target_var):
    path = "/home/monte.flora/machine_learning/main/model_parameters/"
    save_name = join(
        path,
        f'best_{model_name}_{target_var}_params.pkl',
    )
    with open(save_name, "rb") as pkl_file:
        params = pickle.load(pkl_file)
    return params


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

        print(data['training']['examples'].values.shape)


        # Load params
        params = return_params(model_name, target)

        print(params)
        ml.model_name = model_name
        if model_name == 'LogisticRegression':
            data = ml.normalize(data, '')
        
        ml.fit(params=params, data=data)

        fname = f'{model_name}_{target}.pkl'

        print(f'Saving {join(path, fname)}...')
        dump(ml.clf, join(path, fname))




