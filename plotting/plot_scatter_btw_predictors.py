from os.path import join 
from wofs.util import config
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [10,10]

model_name = 'RandomForest'
target = 'matched_to_tornado_0km'

df = pd.read_pickle(f'/work/mflora/ML_DATA/DATA/operational_training_first_hour_resampled_to_{target}_dataset.pkl')

x = 'comp_dz_time_max_ens_std_of_90th'
y = 'hail_time_std_ens_mean_of_90th'
c = target

df.plot.scatter(x=x, y=y, c=c, cmap='seismic', alpha=0.8)

#x_values = df[x].values[::3]
#y_values = df[y].values[::3]
#c_values = df[c].values[::3]


#plt.figure(figsize=(10,10))
#plt.scatter(x_values, y_values, c=c_values, cmap='seismic')
plt.savefig(f'scatter_plot_{x}_{y}_{c}.png', bbox_inches='tight')



