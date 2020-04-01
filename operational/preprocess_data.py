import sys
sys.path.append('/home/monte.flora/machine_learning/build_model')
from PreProcess import PreProcess
import pandas as pd

pp = PreProcess()

for target in ['matched_to_tornado_0km', 'matched_to_severe_wind_0km', 'matched_to_severe_hail_0km']:
    data = pp.load_dataframe(
                target_var_name=target,
                load_filename_dict = {'training': '/work/mflora/ML_DATA/DATA/full_operational_training_first_hour_dataset.pkl'},
                additional_vars_to_drop = [ ],
                )

    data = pp._imputer(data)

    data[pp.TRAINING][pp.EXAMPLES] = pd.concat(
                    [
                        data[pp.TRAINING][pp.EXAMPLES],
                        data[pp.TRAINING][pp.TARGETS]
                    ],
                    axis=1,
                    sort=False
                )

    data = pp.resample(data)

    data[pp.TRAINING][pp.EXAMPLES].to_pickle(f'/work/mflora/ML_DATA/DATA/operational_training_first_hour_resampled_to_{target}_dataset.pkl')


