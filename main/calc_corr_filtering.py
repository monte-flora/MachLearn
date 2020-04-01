import sys
sys.path.append('/home/monte.flora/machine_learning/build_model')
from MachLearn import MachLearn
import pandas as pd

filename = '/work/mflora/ML_DATA/DATA/whole_dataset.pkl'
df = pd.read_pickle(filename)

model = MachLearn(preload=False)
model.correlated_features_to_remove(df=df)

