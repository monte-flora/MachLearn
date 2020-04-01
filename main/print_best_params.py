import pickle
from os.path import join
import os

path = '/home/monte.flora/machine_learning/main/model_parameters'
files = os.listdir(path)
for f in files:
    filename = join(path,f)
    with open(filename, 'rb') as pkl_file:
        best_params = pickle.load(pkl_file)
        print('\n Best Parameters for {}: {}'.format(filename, best_params))
