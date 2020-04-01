import sys
sys.path.append('/home/monte.flora/machine_learning/build_model')
from MachLearn import MachLearn
import pandas as pd
import numpy as np

filename = '/work/mflora/ML_DATA/DATA/whole_dataset.pkl'
df = pd.read_pickle(filename)

print(f'Total number of examples: {len(df)}')

a = df.to_numpy(dtype=float)
print(a.shape)

print( np.sum(np.isnan(a)) )


"""
model = MachLearn( preload=False )
targets = [ 
            'matched_to_tornado_0km',
            'matched_to_severe_wind_0km',
            'matched_to_severe_hail_0km'
            ]

for verify_var in targets:
    model.calc_climo( df=df, verify_var=verify_var)
"""
